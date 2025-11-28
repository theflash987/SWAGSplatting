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
        # If there are no semantic regions for this image, return a zero tensor
        return torch.zeros(gaussian_2d_coords_gpu.shape[0], clip_model.get_sentence_embedding_dimension(), 
                           device=gaussian_2d_coords_gpu.device)

    boxes = torch.tensor([item[0] for item in f_regions_with_box], 
                         dtype=torch.float32, device=gaussian_2d_coords_gpu.device)  # Shape: [M, 4]
    f_regions = torch.stack([item[1] for item in f_regions_with_box]).to(gaussian_2d_coords_gpu.device) # Shape: [M, D]

    # gaussian_2d_coords_gpu: [N, 2] -> [N, 1, 2]
    # boxes: [M, 4] -> [1, M, 4]
    coords_expanded = gaussian_2d_coords_gpu.unsqueeze(1)
    boxes_expanded = boxes.unsqueeze(0)

    # Check if each point is inside each box
    # coords_expanded[..., 0] is the x-coordinate of all points, shape [N, 1]
    # boxes_expanded[..., 0] is the x1 of all boxes, shape [1, M]
    in_box_x = (coords_expanded[..., 0] >= boxes_expanded[..., 0]) & (coords_expanded[..., 0] <= boxes_expanded[..., 2])
    in_box_y = (coords_expanded[..., 1] >= boxes_expanded[..., 1]) & (coords_expanded[..., 1] <= boxes_expanded[..., 3])
    
    # in_box will be a boolean tensor of shape [N, M]
    # in_box[i, j] is True if gaussian i is inside box j
    in_box = in_box_x & in_box_y

    # Find the index of the first box that contains each point
    match_indices = torch.argmax(in_box.int(), dim=1)
    
    # Assign features to each point
    matched_mask = in_box.any(dim=1)
    
    # Use advanced indexing to get the corresponding features from f_regions
    assigned_features = f_regions[match_indices]

    # 6. Set their features to zero if they did not match any box
    # assigned_features has shape [N, D]
    assigned_features[~matched_mask] = 0.0
    
    return assigned_features

def load_segmentation_and_precompute_embeddings(json_path):
    """
    Loads segmentation data and pre-computes text embeddings for all unique labels.
    """
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # Collect all unique labels
    all_labels = sorted(list(set(entry["label"] for entry in raw_data)))
    
    # 2. Encode all unique labels at once -- Pre-computation step
    print(f"Encoding {len(all_labels)} unique labels with CLIP...")
    embeddings = clip_model.encode(all_labels, convert_to_tensor=True, show_progress_bar=True)
    
    label_to_embedding = {label: emb for label, emb in zip(all_labels, embeddings)}
    print("Encoding complete.")

    image_dict = {}
    for entry in raw_data:
        image_name = os.path.basename(entry["image"])
        if image_name not in image_dict:
            image_dict[image_name] = []
        
        image_dict[image_name].append({
            "embedding": label_to_embedding[entry["label"]],
            "box": entry["box"]
        })
    return image_dict


def get_f_region_for_image_precomputed(image_name, seg_data_with_embeddings):

    img_key = os.path.basename(image_name) + ".jpg"
    if img_key not in seg_data_with_embeddings:
        return []

    f_regions = []
    for region in seg_data_with_embeddings[img_key]:
        f_regions.append((region["box"], region["embedding"]))
    return f_regions