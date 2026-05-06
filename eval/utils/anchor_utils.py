import torch
from einops import rearrange
from utils.mathConvert import (
    compute_3d_box_vertices_withlist,
    compute_3d_box_vertices,
    compute_rotation_matrix_from_directions,
    project_3d_to_2d,
    point_to_xxyyzz,
    ROI_3d
)
def generate_center_on_regions(space_range, num_regions):
    """
    Generate center points for each region of a split 3D space.

    Arguments:
    space_range -- tuple of six floats, the min and max coordinates of the 3D space [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    num_regions -- tuple of three integers, the number of regions in the x, y, and z directions

    Returns:
    centers -- torch.Tensor of shape [R^3,3], containing R^3 center points represented as (center_x, center_y, center_z)
    """
    x_min, y_min, z_min = space_range[0]
    x_max, y_max, z_max = space_range[1]
    region_size = ((x_max - x_min) / num_regions[0],
                   (y_max - y_min) / num_regions[1],
                   (z_max - z_min) / num_regions[2])

    centers = torch.empty((0, 3), dtype=torch.float32)
    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            for k in range(num_regions[2]):
                center_x = x_min + (i + 0.5) * region_size[0]
                center_y = y_min + (j + 0.5) * region_size[1]
                center_z = z_min + (k + 0.5) * region_size[2]
                centers = torch.cat([centers, torch.tensor([[center_x, center_y, center_z]], dtype=torch.float32)], dim=0)

    return centers




def generate_anchor_boxes_on_regions(image_size, 
                                     num_regions, 
                                     base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32),
                                     aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32),
                                     dtype=torch.float32, 
                                     device='cpu'):
    """
    Generate a set of anchor boxes with different sizes and aspect ratios for each region of a split image.

    Arguments:
    image_size -- tuple of two integers, the height and width of the original image
    num_regions -- tuple of two integers, the number of regions in the height and width directions
    aspect_ratios -- torch.Tensor of shape [M], containing M aspect ratios for each base size
    dtype -- the data type of the output tensor
    device -- the device of the output tensor

    Returns:
    anchor_boxes -- torch.Tensor of shape [R^2*N*M,4], containing R^2*N*M anchor boxes represented as (center_h, center_w, box_h, box_w)
    """

    # Calculate the base sizes for each region
    region_size = (image_size[0] / num_regions[0], image_size[1] / num_regions[1])

    # Calculate the anchor boxes for each region
    anchor_boxes = torch.empty((0, 4), dtype=dtype, device=device)
    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            center_h = (i + 0.5) * region_size[0]
            center_w = (j + 0.5) * region_size[1]
            base_boxes = generate_anchor_boxes(base_sizes, aspect_ratios, dtype=dtype, device=device)
            base_boxes[:, 0] += center_h
            base_boxes[:, 1] += center_w
            anchor_boxes = torch.cat([anchor_boxes, base_boxes], dim=0)

    return anchor_boxes


def generate_anchor_boxes(base_sizes, aspect_ratios, dtype=torch.float32, device='cpu'):
    """
    Generate a set of anchor boxes with different sizes and aspect ratios.

    Arguments:
    base_sizes -- torch.Tensor of shape [N,2], containing N base sizes for the anchor boxes
    aspect_ratios -- torch.Tensor of shape [M], containing M aspect ratios for each base size
    dtype -- the data type of the output tensor
    device -- the device of the output tensor

    Returns:
    anchor_boxes -- torch.Tensor of shape [N*M,4], containing N*M anchor boxes represented as (center_h, center_w, box_h, box_w)
    """

    num_base_sizes = base_sizes.shape[0]
    num_aspect_ratios = aspect_ratios.shape[0]

    # Generate base anchor boxes
    base_boxes = torch.zeros((num_base_sizes * num_aspect_ratios, 4), dtype=dtype, device=device)
    for i in range(num_base_sizes):
        for j in range(num_aspect_ratios):
            w = torch.sqrt(base_sizes[i, 0] * base_sizes[i, 1] / aspect_ratios[j])
            h = aspect_ratios[j] * w
            idx = i * num_aspect_ratios + j
            base_boxes[idx] = torch.tensor([0, 0, h, w], dtype=dtype, device=device)

    return base_boxes




def assign_labels(center_points, gt_center, distance_threshold=0.3, topk=5):
    """
    Assign labels to a set of bounding box proposals based on their IoU with ground truth boxes.

    Arguments:
    center_points -- torch.Tensor of shape [B,T,N,3], representing the center points of the proposals for each frame in each clip
    gt_center -- torch.Tensor of shape [B,T,3], representing the ground truth centers for each frame in each clip
    distance_threshold -- float, the distance threshold for a proposal to be considered a positive match with a ground truth box

    Returns:
    labels -- torch.Tensor of shape [B,T,N], containing the assigned labels for each proposal (0 for background, 1 for object)
    """
    center_points = center_points.detach()
    gt_center = gt_center.detach()

    b,t = gt_center.shape[:2]    #[B,T,N,3]

    # Calculate the distance between each proposal and the ground truth box
    distance = calculate_distance(center_points.view(-1, center_points.shape[-2], center_points.shape[-1]),   # [B*T,N,3]
                        gt_center.view(-1, gt_center.shape[-1]))                    # [B*T,3] -> [B*T,N]
    distance = distance.view(center_points.shape[:-1])    # [B,T,N]

    # Assign labels to the proposals based on their distance with the ground truth box
    labels = distance < distance_threshold

    if not labels.any():
        labels = process_labels(labels, distance, topk)

    return labels


def calculate_iou(boxes1, boxes2):
    """
    Calculate the IoU between two sets of bounding boxes using AABB approximation.

    Arguments:
    boxes1 -- torch.Tensor of shape [..., 9], bounding boxes as [x, y, z, l, w, h, roll, pitch, yaw]
    boxes2 -- torch.Tensor of shape [..., 9], same structure as boxes1

    Returns:
    iou -- torch.Tensor of shape [...], IoU for each corresponding box pair
    """
    vert1 = compute_3d_box_vertices_withlist(boxes1)  # Shape: [..., 8, 3]
    vert2 = compute_3d_box_vertices_withlist(boxes2)  # Shape: [..., 8, 3]

    # Flatten leading dimensions for batch processing
    orig_shape = vert1.shape[:-2]  # Save original leading dimensions
    vert1_flat = vert1.view(-1, 8, 3)  # [B, 8, 3]
    vert2_flat = vert2.view(-1, 8, 3)  # [B, 8, 3]

    # Compute AABB for each box
    aabb_min1 = torch.amin(vert1_flat, dim=1)  # [B, 3]
    aabb_max1 = torch.amax(vert1_flat, dim=1)  # [B, 3]
    aabb_min2 = torch.amin(vert2_flat, dim=1)  # [B, 3]
    aabb_max2 = torch.amax(vert2_flat, dim=1)  # [B, 3]

    # Calculate intersection volume
    inter_min = torch.maximum(aabb_min1, aabb_min2)
    inter_max = torch.minimum(aabb_max1, aabb_max2)
    inter_dims = torch.clamp(inter_max - inter_min, min=0)
    inter_volume = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]  # [B]

    # Calculate individual volumes
    vol1 = (aabb_max1[:, 0] - aabb_min1[:, 0]) * \
           (aabb_max1[:, 1] - aabb_min1[:, 1]) * \
           (aabb_max1[:, 2] - aabb_min1[:, 2])
    vol2 = (aabb_max2[:, 0] - aabb_min2[:, 0]) * \
           (aabb_max2[:, 1] - aabb_min2[:, 1]) * \
           (aabb_max2[:, 2] - aabb_min2[:, 2])
    
    union_volume = vol1 + vol2 - inter_volume  # [B]
    iou_flat = inter_volume / torch.clamp(union_volume, min=1e-6)  # [B]

    # Restore original leading dimensions
    iou = iou_flat.view(orig_shape)  # Shape: [...]
    return iou

def calculate_distance(center_points, gt_center):
    """
    Calculate the Euclidean distance between center points and ground truth centers.

    Arguments:
    center_points -- torch.Tensor of shape [BT,N,3], representing the center points of the proposals for each frame in each clip
    gt_center -- torch.Tensor of shape [BT,3], representing the ground truth centers for each frame in each clip

    Returns:
    distance -- torch.Tensor of shape [BT,N], containing the Euclidean distance between each center point and the ground truth center
    """
    # Expand gt_center to match the shape of center_points
    gt_center_expanded = gt_center.unsqueeze(1)  # shape: [BT,1,3]
    distance = torch.norm(center_points - gt_center_expanded, dim=-1)  # shape: [BT,N]
    return distance

def process_labels(labels, distance, topk=10):
    '''
    labels: in shape [B,T,N], bool
    distance: in shape [B,T,N]
    '''
    B,T,N = labels.shape

    labels = rearrange(labels, 'b t n -> (b t n)')
    distance = rearrange(distance, 'b t n -> (b t n)')

    if not labels.any():
        # no pos assigned, choose topk anchors with smallest distance as positives
        _, topk_indices = torch.topk(distance, k=topk, largest=False)
        labels[topk_indices] = True
    
    labels = rearrange(labels, '(b t n) -> b t n', b=B, t=T, n=N)
    return labels