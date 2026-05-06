from typing import List, Dict
import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import numpy as np
import torch
import math
from evaluation.structures import BBox, ResponseTrack, BBox3D
# from pytorch3d.ops import box3d_overlap
from utils.mathConvert import (
    compute_3d_box_vertices,
    compute_3d_box_vertices_withlist,
    compute_rotation_matrix_from_directions,
    project_3d_to_2d,
)
from mmcv.ops import box_iou_rotated  # Ensure mmcv-full is installed
PRINT_FORMAT = "{:<30s} {:<15s}"
def calculate_iou_7dof(
    boxes1,
    boxes2,
    return_giou=False,
    angles_in_degrees=False,
    eps=1e-6,
    maxchunk=32768,
    maxchunk_m=None,
):
    """
    7DoF IoU/GIoU for 3D oriented boxes (yaw only), with optional chunked computation.

    Supported box formats:
      - 9D: [x, y, z, l, w, h, roll, pitch, yaw]
      - 7D: [x, y, z, l, w, h, yaw]

    Assumptions:
      - Coordinate system: x (forward+), y (left+), z (up+).
      - GIoU uses axis-aligned AABB of BEV projections as enclosing volume (stable, practical).
      - roll/pitch are ignored; only yaw is used.

    Returns:
      - If boxes1 is (B,N,D) and boxes2 is (B,M,D): IoU shape (B,N) when M==1 else (B,N,M)
      - If boxes1 is (N,D) and boxes2 is (N,D): elementwise IoU shape (N,1)  [与 GiouLoss 兼容]
      - If boxes1 is (N,D) and boxes2 is (M,D): IoU shape (N,M)
      When return_giou=True, returns (iou, giou) with the same shape.
    """
    assert boxes1.dim() in (2, 3), "boxes1 must be (N,D) or (B,N,D)"
    D = boxes1.size(-1)
    assert D in (7, 9), "Only 7D or 9D boxes are supported"

    device = boxes1.device
    dtype = boxes1.dtype
    # yaw 索引
    idx_x, idx_y, idx_z = 0, 1, 2
    idx_l, idx_w, idx_h = 3, 4, 5
    idx_yaw = 8 if D == 9 else 6
    # 正常批处理路径
    # 规范到 (B,N,D) 和 (B,M,D)
    if boxes1.dim() == 2:
        B = 1
        boxes1_b = boxes1.unsqueeze(0)
    else:
        B = boxes1.size(0)
        boxes1_b = boxes1

    if boxes2.dim() == 1:
        boxes2_b = boxes2.unsqueeze(0).unsqueeze(1).expand(B, 1, D)
    elif boxes2.dim() == 2:
        if boxes2.size(0) == B:
            boxes2_b = boxes2.unsqueeze(1)  # (B,1,D)
        else:
            boxes2_b = boxes2.unsqueeze(0).expand(B, -1, -1)  # (B,M,D)
    elif boxes2.dim() == 3:
        assert boxes2.size(0) == B, "boxes2 batch dim must match boxes1"
        boxes2_b = boxes2
    else:
        raise AssertionError("boxes2 must be (D), (M,D), (B,D) or (B,M,D)")

    B, N, _ = boxes1_b.shape
    M = boxes2_b.shape[1]
    cn = max(1, int(maxchunk)) if maxchunk is not None else N
    cm = max(1, int(maxchunk_m)) if (maxchunk_m is not None and M > 1) else M

    ious_b = []
    gious_b = [] if return_giou else None

    for b in range(B):
        b1 = boxes1_b[b]  # (N,D)
        b2 = boxes2_b[b]  # (M,D)

        x1 = b1[:, idx_x]; y1 = b1[:, idx_y]; z1 = b1[:, idx_z]
        l1 = b1[:, idx_l].clamp_min(eps); w1 = b1[:, idx_w].clamp_min(eps); h1 = b1[:, idx_h].clamp_min(eps)
        yaw1 = b1[:, idx_yaw]
        x2 = b2[:, idx_x]; y2 = b2[:, idx_y]; z2 = b2[:, idx_z]
        l2 = b2[:, idx_l].clamp_min(eps); w2 = b2[:, idx_w].clamp_min(eps); h2 = b2[:, idx_h].clamp_min(eps)
        yaw2 = b2[:, idx_yaw]

        if angles_in_degrees:
            yaw1_deg_all = yaw1
            yaw2_deg_all = yaw2
            yaw1_rad_all = yaw1 * (math.pi / 180.0)
            yaw2_rad_all = yaw2 * (math.pi / 180.0)
        else:
            yaw1_deg_all = yaw1 * (180.0 / math.pi)
            yaw2_deg_all = yaw2 * (180.0 / math.pi)
            yaw1_rad_all = yaw1
            yaw2_rad_all = yaw2

        iou_3d_b = torch.empty((N, M), device=device, dtype=dtype)
        giou_3d_b = torch.empty((N, M), device=device, dtype=dtype) if return_giou else None

        for ns in range(0, N, cn):
            ne = min(ns + cn, N)

            x1_s = x1[ns:ne]; y1_s = y1[ns:ne]; z1_s = z1[ns:ne]
            l1_s = l1[ns:ne]; w1_s = w1[ns:ne]; h1_s = h1[ns:ne]
            yaw1_deg = yaw1_deg_all[ns:ne]
            yaw1_rad = yaw1_rad_all[ns:ne]

            bottom1 = (z1_s - h1_s / 2.0).unsqueeze(-1)  # (nc,1)
            top1    = (z1_s + h1_s / 2.0).unsqueeze(-1)  # (nc,1)
            area1   = (l1_s * w1_s).unsqueeze(-1)        # (nc,1)

            boxes1_2d = torch.stack([x1_s, y1_s, l1_s, w1_s, yaw1_deg], dim=-1).to(device=device, dtype=torch.float32)

            for ms in range(0, M, cm) if M > 1 else [0]:
                me = min(ms + cm, M)

                x2_s = x2[ms:me]; y2_s = y2[ms:me]; z2_s = z2[ms:me]
                l2_s = l2[ms:me]; w2_s = w2[ms:me]; h2_s = h2[ms:me]
                yaw2_deg = yaw2_deg_all[ms:me]
                yaw2_rad = yaw2_rad_all[ms:me]

                boxes2_2d = torch.stack([x2_s, y2_s, l2_s, w2_s, yaw2_deg], dim=-1).to(device=device, dtype=torch.float32)

                iou_2d = box_iou_rotated(boxes1_2d, boxes2_2d).to(dtype)             # (nc,mc)
                area2  = (l2_s * w2_s).unsqueeze(-2)                                  # (1,mc)
                inter_area_2d = iou_2d * (area1 + area2) / (1.0 + iou_2d + eps)       # (nc,mc)

                bottom2 = (z2_s - h2_s / 2.0).unsqueeze(-2)                           # (1,mc)
                top2    = (z2_s + h2_s / 2.0).unsqueeze(-2)                           # (1,mc)

                inter_bottom = torch.maximum(bottom1, bottom2)
                inter_top    = torch.minimum(top1, top2)
                inter_height = torch.clamp(inter_top - inter_bottom, min=0.0)

                vol1 = area1 * h1_s.unsqueeze(-1)                                     # (nc,1)
                vol2 = area2 * h2_s.unsqueeze(-2)                                     # (1,mc)

                inter_volume = inter_area_2d * inter_height
                union_volume = vol1 + vol2 - inter_volume
                iou_3d_s = torch.where(
                    union_volume > 0,
                    inter_volume / (union_volume + eps),
                    inter_volume.new_zeros(()).expand_as(union_volume),
                )

                iou_3d_b[ns:ne, ms:me] = iou_3d_s

                if return_giou:
                    # 解析式 AABB 半轴（BEV），避免角点广播
                    c1 = torch.abs(torch.cos(yaw1_rad)).unsqueeze(-1)  # (nc,1)
                    s1 = torch.abs(torch.sin(yaw1_rad)).unsqueeze(-1)
                    c2 = torch.abs(torch.cos(yaw2_rad)).unsqueeze(0)   # (1,mc)
                    s2 = torch.abs(torch.sin(yaw2_rad)).unsqueeze(0)

                    half_wx1 = 0.5 * (l1_s.unsqueeze(-1) * c1 + w1_s.unsqueeze(-1) * s1)  # (nc,1)
                    half_wy1 = 0.5 * (l1_s.unsqueeze(-1) * s1 + w1_s.unsqueeze(-1) * c1)  # (nc,1)
                    half_wx2 = 0.5 * (l2_s.unsqueeze(0) * c2 + w2_s.unsqueeze(0) * s2)    # (1,mc)
                    half_wy2 = 0.5 * (l2_s.unsqueeze(0) * s2 + w2_s.unsqueeze(0) * c2)    # (1,mc)

                    minx1 = x1_s.unsqueeze(-1) - half_wx1; maxx1 = x1_s.unsqueeze(-1) + half_wx1  # (nc,1)
                    miny1 = y1_s.unsqueeze(-1) - half_wy1; maxy1 = y1_s.unsqueeze(-1) + half_wy1  # (nc,1)
                    minx2 = x2_s.unsqueeze(0)  - half_wx2; maxx2 = x2_s.unsqueeze(0)  + half_wx2  # (1,mc)
                    miny2 = y2_s.unsqueeze(0)  - half_wy2; maxy2 = y2_s.unsqueeze(0)  + half_wy2  # (1,mc)

                    minx = torch.minimum(minx1, minx2)  # (nc,mc)
                    maxx = torch.maximum(maxx1, maxx2)
                    miny = torch.minimum(miny1, miny2)
                    maxy = torch.maximum(maxy1, maxy2)

                    aabb_w = torch.clamp(maxx - minx, min=0.0)
                    aabb_h = torch.clamp(maxy - miny, min=0.0)
                    area_C2D = aabb_w * aabb_h                                     # (nc,mc)

                    min_bottom = torch.minimum(bottom1, bottom2)
                    max_top    = torch.maximum(top1, top2)
                    C_height   = torch.clamp(max_top - min_bottom, min=0.0)

                    C_vol = area_C2D * C_height + eps
                    giou_3d_s = iou_3d_s - (C_vol - union_volume) / C_vol

                    giou_3d_b[ns:ne, ms:me] = giou_3d_s

        if M == 1:
            ious_b.append(iou_3d_b.squeeze(-1))
            if return_giou:
                gious_b.append(giou_3d_b.squeeze(-1))
        else:
            ious_b.append(iou_3d_b)
            if return_giou:
                gious_b.append(giou_3d_b)

    iou_3d = torch.stack(ious_b, dim=0)   # (B,N) or (B,N,M)
    if boxes1.dim() == 2:
        iou_3d = iou_3d.squeeze(0)        # (N,) or (N,M)

    if return_giou:
        giou_3d = torch.stack(gious_b, dim=0)
        if boxes1.dim() == 2:
            giou_3d = giou_3d.squeeze(0)
        return iou_3d, giou_3d

    # 检查 NaN
    if torch.isnan(iou_3d).any():
        raise ValueError("NaN detected in IoU computation")
    # 返回
    return iou_3d


def segment_iou(
    target_segment: np.ndarray, candidate_segments: np.ndarray
) -> np.ndarray:
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1 + 1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0] + 1)
        + (target_segment[1] - target_segment[0] + 1)
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def spatial_iou(box1: BBox3D, box2: BBox3D) -> float:
    """
    Computes iou between two bounding boxes
    """
    # vert1 = compute_3d_box_vertices_withlist(box1.to_tensor_list())
    # vert2 = compute_3d_box_vertices_withlist(box2.to_tensor_list())
    # if vert1.dim() == 2:
    #     vert1 = vert1.unsqueeze(0)
    # if vert2.dim() == 2:
    #     vert2 = vert2.unsqueeze(0)
    # _, iou = box3d_overlap(vert1, vert2) # [1, 1]
    # iou = iou.item()
    iou = calculate_iou_7dof(
        box1.to_7d_tensor_list().clone().detach().unsqueeze(0),
        box2.to_7d_tensor_list().clone().detach().unsqueeze(0),
    ).item()

    return iou


def spatial_intersection(box1: BBox3D, box2: BBox3D) -> float:
    """
    Computes intersection between two bounding boxes
    """
    # vert1 = compute_3d_box_vertices_withlist(box1.to_tensor_list())
    # vert2 = compute_3d_box_vertices_withlist(box2.to_tensor_list())
    # if vert1.dim() == 2:
    #     vert1 = vert1.unsqueeze(0)
    # if vert2.dim() == 2:
    #     vert2 = vert2.unsqueeze(0)
    # inter, _ = box3d_overlap(vert1, vert2) # [1, 1]
    # inter = inter.item()
    iou = calculate_iou_7dof(
        box1.to_7d_tensor_list().clone().detach().unsqueeze(0),
        box2.to_7d_tensor_list().clone().detach().unsqueeze(0),
    )
    vol1 = box1.volume()
    vol2 = box2.volume()
    inter = iou.item() * (vol1 + vol2) / (1 + iou.item() + 1e-6)

    return inter


def spatio_temporal_iou_response_track(rt1: ResponseTrack, rt2: ResponseTrack) -> float:
    """
    Computes tube-iou between two response track windows.
    Note: This assumes that each bbox in the list corresponds to a different
    frame. Cannot handle multiple bboxes per frame.

    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics
    """
    # Map frame numbers to boxes
    boxes1_dict = {box.fno: box for box in rt1.bboxes}
    inter = 0.0
    # hypervolume1 = 1e-6
    # hypervolume2 = 1e-6
    # Find matching frame boxes and estimate iou
    for box2 in rt2.bboxes:
        box1 = boxes1_dict.get(box2.fno, None)
        if box1 is None:
            continue
        inter += spatial_intersection(box1, box2)
        # hypervolume1 += box1.volume()
        # hypervolume2 += box2.volume()
    # Find overall volume of the two respose tracks
    hypervolume1 = rt1.hypervolume()
    hypervolume2 = rt2.hypervolume()

    iou = inter / (hypervolume1 + hypervolume2 - inter)

    return iou

def spatio_temporal_mate_response_track(
    rt1: ResponseTrack, rt2: ResponseTrack
) -> float:
    """
    Computes tube-mate between two response track windows.
    Note: This assumes that each bbox in the list corresponds to a different
    frame. Cannot handle multiple bboxes per frame.
    """
    # Map frame numbers to boxes
    boxes1_dict = {box.fno: box for box in rt1.bboxes}
    total_distance = 0.0
    count = 0
    # Find matching frame boxes and estimate distance
    for box2 in rt2.bboxes:
        box1 = boxes1_dict.get(box2.fno, None)
        if box1 is None:
            continue
        center1 = box1.center()
        center2 = box2.center()
        distance = np.linalg.norm(
            np.array(center1) - np.array(center2)
        )
        total_distance += distance
        count += 1
    if count == 0:
        return float('inf')  # No overlapping frames

    mate = total_distance / count

    return mate

def spatio_temporal_iou(
    target_rt: ResponseTrack, candidate_rts: List[ResponseTrack]
) -> np.ndarray:
    """
    Computes spatio-temporal IoU between a target response track (prediction) and
    multiple candidate response tracks (ground-truth).
    """
    ious = []
    for candidate_rt in candidate_rts:
        ious.append(spatio_temporal_iou_response_track(target_rt, candidate_rt))

    return np.array(ious)

def spatio_temporal_mate(
    target_rt: ResponseTrack, candidate_rts: List[ResponseTrack]
) -> np.ndarray:
    """
    Computes spatio-temporal MATE between a target response track (prediction) and
    multiple candidate response tracks (ground-truth).
    """
    mates = []
    for candidate_rt in candidate_rts:
        mates.append(spatio_temporal_mate_response_track(target_rt, candidate_rt))

    return np.array(mates)

# Tracking related utils


def spatial_matches_response_track(
    pred: ResponseTrack, gt: ResponseTrack
) -> Dict[str, float]:
    """
    For each bounding box in gt, find a match in pred and measure the per-frame IoU.
    Set IoU to zero if no match is found.

    Note: This assumes that each bbox in the list corresponds to a different
    frame. Cannot handle multiple bboxes per frame.
    """
    # Map frame numbers to boxes
    gt_dict = {box.fno: box for box in gt.bboxes}
    ious = {box.fno: 0.0 for box in gt.bboxes}
    # Find matching frame boxes and estimate iou
    for pred_box in pred.bboxes:
        gt_box = gt_dict.get(pred_box.fno, None)
        if gt_box is not None:
            ious[gt_box.fno] = spatial_iou(gt_box, pred_box)
    return ious


def spatio_temporal_iou_matches(
    target_rt: ResponseTrack,
    candidate_rts: List[ResponseTrack],
) -> List[Dict[str, float]]:
    """
    For each BBox in each candidate response track (ground-truth),
    find the IoU b/w itself and a BBox from the target response track (prediction).
    In case no match is found for a particular BBox in the candidate,
    then the IoU is set to zero.
    """
    ious = []
    for candidate_rt in candidate_rts:
        ious.append(spatial_matches_response_track(target_rt, candidate_rt))
    return ious
