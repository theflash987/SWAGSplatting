import json
import torch
import torch.nn as nn
import os
from sentence_transformers import SentenceTransformer

clip_model = SentenceTransformer('clip-ViT-L-14')

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x >= x1 and x <= x2 and y >= y1 and y <= y2


def assign_f_region_to_gaussians_vectorized(gaussian_2d_coords_gpu, f_regions_with_box):
    if not f_regions_with_box:
        # 如果这张图没有任何语义区域，返回一个零张量
        return torch.zeros(gaussian_2d_coords_gpu.shape[0], clip_model.get_sentence_embedding_dimension(), 
                           device=gaussian_2d_coords_gpu.device)

    # 1. 将所有边界框和特征向量堆叠成张量
    boxes = torch.tensor([item[0] for item in f_regions_with_box], 
                         dtype=torch.float32, device=gaussian_2d_coords_gpu.device)  # Shape: [M, 4]
    f_regions = torch.stack([item[1] for item in f_regions_with_box]).to(gaussian_2d_coords_gpu.device) # Shape: [M, D]

    # 2. 扩展维度以进行广播操作
    # gaussian_2d_coords_gpu: [N, 2] -> [N, 1, 2]
    # boxes: [M, 4] -> [1, M, 4]
    coords_expanded = gaussian_2d_coords_gpu.unsqueeze(1)
    boxes_expanded = boxes.unsqueeze(0)

    # 3. 并行化检查每个点是否在每个框内
    # coords_expanded[..., 0] is the x-coordinate of all points, shape [N, 1]
    # boxes_expanded[..., 0] is the x1 of all boxes, shape [1, M]
    in_box_x = (coords_expanded[..., 0] >= boxes_expanded[..., 0]) & (coords_expanded[..., 0] <= boxes_expanded[..., 2])
    in_box_y = (coords_expanded[..., 1] >= boxes_expanded[..., 1]) & (coords_expanded[..., 1] <= boxes_expanded[..., 3])
    
    # in_box will be a boolean tensor of shape [N, M]
    # in_box[i, j] is True if gaussian i is inside box j
    in_box = in_box_x & in_box_y

    # 4. 找到每个点匹配的第一个框的索引
    # `argmax` 会返回第一个True（即最大值）的索引。
    # 如果某一行全为False, argmax会返回0。我们需要一个掩码来处理这种情况。
    match_indices = torch.argmax(in_box.int(), dim=1)
    
    # 5. 为每个点分配特征
    # 使用 `in_box.any(dim=1)` 来检查每个点是否至少匹配一个框
    matched_mask = in_box.any(dim=1)
    
    # 使用高级索引从 f_regions 中获取对应的特征
    assigned_features = f_regions[match_indices]

    # 6. 对于没有匹配任何框的点，将其特征置为零
    # assigned_features 的维度是 [N, D]
    assigned_features[~matched_mask] = 0.0
    
    return assigned_features

def load_segmentation_and_precompute_embeddings(json_path):
    """
    Loads segmentation data and pre-computes text embeddings for all unique labels.
    """
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # 1. 收集所有唯一的标签
    all_labels = sorted(list(set(entry["label"] for entry in raw_data)))
    
    # 2. 一次性编码所有唯一标签
    print(f"Encoding {len(all_labels)} unique labels with CLIP...")
    # 使用批量编码以提高效率
    embeddings = clip_model.encode(all_labels, convert_to_tensor=True, show_progress_bar=True)
    
    label_to_embedding = {label: emb for label, emb in zip(all_labels, embeddings)}
    print("Encoding complete.")

    # 3. 构建新的数据结构
    image_dict = {}
    for entry in raw_data:
        image_name = os.path.basename(entry["image"])
        if image_name not in image_dict:
            image_dict[image_name] = []
        
        # 直接存储预计算好的嵌入向量
        image_dict[image_name].append({
            "embedding": label_to_embedding[entry["label"]],
            "box": entry["box"]
        })
    return image_dict


def get_f_region_for_image_precomputed(image_name, seg_data_with_embeddings):
    """
    Retrieves region data (box and pre-computed embedding) for an image.
    """
    img_key = os.path.basename(image_name) + ".jpg"
    if img_key not in seg_data_with_embeddings:
        # 如果训练/测试图像不在json里，返回空列表是安全的
        return []

    f_regions = []
    for region in seg_data_with_embeddings[img_key]:
        # 不再需要在这里调用 model.encode
        f_regions.append((region["box"], region["embedding"]))
    return f_regions