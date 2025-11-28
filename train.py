#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from random import randint
import random
import numpy as np
from utils.loss_utils import l1_loss, l2_loss, ssim, Channel_wise_depth_consistency, GrayWorldAssumptionLoss, EdgeSmoothnessLoss
from utils.weight_utils_quick import robust_mask
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn as nn
import gc
from torchvision import models, transforms
from utils.semantic_utils import load_segmentation_and_precompute_embeddings, get_f_region_for_image_precomputed, assign_f_region_to_gaussians_vectorized
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    stage = getattr(opt, 'stage', None)
    finetune_entered = False
    
    # Semantic source path
    seg_json_path = os.path.join(dataset.source_path, "output.json")
    seg_data = load_segmentation_and_precompute_embeddings(seg_json_path)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    uncertainty_optimizer = None
    if hasattr(scene, 'uncertainty_params') and scene.uncertainty_params is not None:
        uncertainty_optimizer = torch.optim.Adam(
            [p for p in scene.uncertainty_params.parameters()], 
            # lr=1e-5 
            lr = 1e-4
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    inlier_threshold = 1.0

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, depth_threshold = opt.depth_threshold * scene.cameras_extent, iteration = iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["depth"]
        
        screenspace_points = render_pkg["viewspace_points"]
        
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        visible_screenspace = screenspace_points[visibility_filter]

        pixel_coords_gpu = torch.stack([
            (visible_screenspace[:, 0] + 1.0) * W * 0.5,
            (visible_screenspace[:, 1] + 1.0) * H * 0.5
        ], dim=1)

        # Depth alignment loss
        depths, t_d, t_b, attn, bs = render_pkg["depths"], render_pkg["transmission_map_d"], render_pkg[
            "backscattering"], render_pkg["attn"], render_pkg["bs"]
        attn = torch.mean(attn, dim=0)
        bs = torch.mean(bs, dim=0)
        L_da = Channel_wise_depth_consistency()
        l_d = L_da(t_d, attn, depths.detach(), t_b, bs)

        # Gray world loss
        L_gw = GrayWorldAssumptionLoss()
        clean = render_pkg["color_clean"]
        l_g = L_gw(clean)

        # Reconstruction Loss
        gt_image = viewpoint_cam.original_image.cuda()
        or_depth = viewpoint_cam.original_depth.cuda()

        # BMM Calculation
        if pipe.BMM_Flag:
            Ll1 = l1_loss(image, gt_image)
            mask, stats, mask_1, mask_2 = robust_mask(Ll1, inlier_threshold)
            inlier_threshold = stats["inlier_threshold"]
        else:
            mask = torch.ones_like(image).float()

        Ll1_depth = torch.mean(l1_loss(mask * depth, mask * or_depth))
        Ll1 = l1_loss(mask * image, mask * gt_image)
        
        Ll2 = l2_loss(mask * image, mask * gt_image)

        # Edge smoothness loss
        L_smooth = EdgeSmoothnessLoss(beta=5)
        smooth_loss = L_smooth(image, depth)
        lambda_smooth = 0.05

        # Learnable uncertainty parameters
        weight, log_uncertainty = scene.get_camera_weight(viewpoint_cam, scale=1.0)

        # Semantic loss
        f_regions = get_f_region_for_image_precomputed(viewpoint_cam.image_name, seg_data)
        f_region = assign_f_region_to_gaussians_vectorized(pixel_coords_gpu.detach(), f_regions)  
        f_region = gaussians.proj(f_region)  
        f_gaussian = gaussians.get_semantic_feature(visibility_filter)
        semantic_loss = torch.mean((f_gaussian - f_region.detach()) ** 2)

        if iteration >= opt.mlp_gradient_stop:
            gaussians.mlp_head.requires_grad_(False)
        if iteration <= opt.densify_from_iter:
            loss = 0.1 * Ll1_depth + l_d + l_g
        else:
            l1_weight = 0.7 + 0.5 * (iteration / opt.iterations)
            # Second stage optimisation
            is_finetune = (stage is not None) and (iteration >= stage)
            
            # Freeze geometry lr to stable the basic property, unfreeze MLP head
            if is_finetune and not finetune_entered:
                # Freeze geometry-related parameters (set lr to 0, minimal intrusion)
                for group in gaussians.optimizer.param_groups:
                    if group.get("name") in ["xyz", "scaling", "rotation"]:
                        group['lr'] = 0.0
                # Unfreeze MLP head to allow fitting color residuals later
                gaussians.mlp_head.requires_grad_(True)
                finetune_entered = True
                print(f"[PSNR Finetune] Entered at iter {iteration}. Frozen geometry lrs, unfroze MLP head.")

            # Emphasize L2 during finetuning, decay structure/regularization/depth/semantic
            if is_finetune:
                l2_weight = 0.9
                lambda_dssim_eff = 0.0
                lambda_smooth_eff = 0.0
                depth_w = 0.0
                semantic_weight = 0.0
            else:
                l2_weight = 0.1 * (iteration / opt.iterations)
                lambda_dssim_eff = opt.lambda_dssim
                lambda_smooth_eff = lambda_smooth
                depth_w = 0.1

            if not is_finetune:
                semantic_weight = max(0.01, 0.3 * (1.0 - iteration / opt.iterations))

            if weight == 1:
                l2_term = torch.mean(Ll2)
                loss = torch.mean(l1_weight * (1.0 - lambda_dssim_eff) * Ll1) \
                       + lambda_dssim_eff * (1.0 - ssim(mask * image, mask * gt_image)) \
                       + l_d + l_g \
                       + depth_w * Ll1_depth \
                       + lambda_smooth_eff * smooth_loss \
                       + l2_weight * l2_term \
                       + semantic_weight * semantic_loss
            else:
                loss = torch.mean(l1_weight * (1.0 - lambda_dssim_eff) * Ll1) \
                       + lambda_dssim_eff * (1.0 - ssim(mask * image, mask * gt_image)) \
                       + l_d + l_g \
                       + depth_w * Ll1_depth \
                       + lambda_smooth_eff * smooth_loss \
                       + l2_weight * l2_term

                if hasattr(opt, 'adaptive') and opt.adaptive:
                    # loss = (weight * loss) / 2 + (opt.alpha * log_uncertainty) / 2
                    loss = (weight * loss) / 2 + log_uncertainty / 2

                # Apply fixed frame weights if --frame flag is used
                if hasattr(opt, 'frame') and opt.frame is not None:
                    loss = weight * loss

        loss.backward()
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer_net.step()
                gaussians.optimizer_net.zero_grad(set_to_none=True)
                gaussians.scheduler_net.step()
                
                if uncertainty_optimizer is not None:
                    uncertainty_optimizer.step()
                    uncertainty_optimizer.zero_grad(set_to_none=True)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"])
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, viewpoint_cam)
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    if hasattr(scene, 'print_interp_weights'):
        print("\n===== Final Interpolated Frame Weights =====")
        scene.print_interp_weights()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', torch.mean(Ll1).item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--stage', type=int, default=None, help='Second stage of the optimisation')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 12_000, 15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 12_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--frame", type=float, default=None, 
                        help="Set initial weight for interpolated frames")
    parser.add_argument("--adaptive", action="store_true", default=False,
                        help="Use learnable uncertainty parameters for interpolated frame weights")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Coefficient for log_uncertainty term in adaptive loss")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_args = lp.extract(args)
    model_args.adaptive = args.adaptive
    model_args.frame = args.frame
    model_args.alpha = args.alpha
    model_args.stage = args.stage

    opt = op.extract(args)
    opt.adaptive = args.adaptive
    opt.alpha = args.alpha
    opt.frame = args.frame
    opt.stage = args.stage
    
    training(model_args, opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
