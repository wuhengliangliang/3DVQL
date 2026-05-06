from collections import OrderedDict
from typing import List, Dict, Tuple

import numpy as np

from metrics.spatio_temporal_metrics import SpatioTemporalDetection
from metrics.success_metrics import SuccessMetrics
from metrics.temporal_metrics import TemporalDetection
from metrics.tracking_metrics import TrackingMetrics
from metrics.MATE_metrics import MATEDetection
from metrics.utils import BBox, ResponseTrack


METRIC_FNS = [
    lambda gt, pred: MATEDetection(gt, pred, ignore_mate_averaging=False).get_metrics(),
    lambda gt, pred: TemporalDetection(gt, pred).get_metrics(),
    lambda gt, pred: SpatioTemporalDetection(gt, pred).get_metrics(),
    lambda gt, pred: TrackingMetrics(gt, pred, ignore_iou_averaging=True).get_metrics(),
    lambda gt, pred: SuccessMetrics(gt, pred, ignore_iou_averaging=True).get_metrics(),
    
]


def compute_visual_query_metrics(
    predicted_response_track: List[List[ResponseTrack]],
    ground_truth_response_track: List[ResponseTrack],
    visual_crop_boxes: List[BBox],
    accessed_frames_in_clip: List[int] = None,
    total_frames_in_clip: List[int] = None,
    area_ranges: Dict[str, List[int]] = {
        "all": [0 ** 2, 1e5 ** 2],
        "small": [0 ** 2, 64 ** 2],
        "medium": [64 ** 2, 192 ** 2],
        "large": [192 ** 2, 1e5 ** 2],
    },
    vc_rt_pairings: Dict[str, Tuple[str, str]] = {
        "all": ("all", "all"),
    },
) -> Dict[str, float]:
    """
    Compute model performance on the visual query task. Includes the following metrics:
        * Temporal AP
        * SpatioTemporal AP
        * MATE AP
        * Success
        * Tracking % recovery
        * Search efficiency
    """

    # Calculate visual-crop volumes
    vc_vols = np.array(
        [
            vc_bbox.w * vc_bbox.l * vc_bbox.h
            for vc_bbox in visual_crop_boxes
        ]
    )
    # Calculate response-track max volumes
    rt_vols = []
    for rt in ground_truth_response_track:
        vol = (
            np.array(
                [
                    rt_bbox.w * rt_bbox.l * rt_bbox.h
                    for rt_bbox in rt.bboxes
                ]
            )
            .max()
            .item()
        )
        rt_vols.append(vol)
    rt_vols = np.array(rt_vols)

    num_valid = 0
    # Calculate metrics for each vc_rt_pairing
    pair_metrics = OrderedDict()
    for pair_name, (vc_cat, rt_cat) in vc_rt_pairings.items():
        vc_range = area_ranges[vc_cat]
        rt_range = area_ranges[rt_cat]
        # Get data points satifying the pairing criterion
        mask = (
            (vc_vols >= vc_range[0])
            & (vc_vols < vc_range[1])
            & (rt_vols >= rt_range[0])
            & (rt_vols < rt_range[1])
        )
        num_valid += mask.sum()
        # Ignore pairing if there are not valid data points
        if np.count_nonzero(mask) == 0:
            continue
        # Calculate metrics
        pred_rt = [predicted_response_track[i] for i, cond in enumerate(mask) if cond]
        gt_rt = [ground_truth_response_track[i] for i, cond in enumerate(mask) if cond]
        if accessed_frames_in_clip is not None:
            acc_frames = [
                accessed_frames_in_clip[i] for i, cond in enumerate(mask) if cond
            ]
            tot_frames = [
                total_frames_in_clip[i] for i, cond in enumerate(mask) if cond
            ]
        metrics = OrderedDict()
        for metric_fn in METRIC_FNS:
            metrics.update(metric_fn(gt_rt, pred_rt))
        if accessed_frames_in_clip is not None and len(acc_frames) > 0:
            metrics["Search efficiency (%)"] = (
                1 - np.array(acc_frames).astype(np.float32) / np.array(tot_frames)
            ).mean() * 100.0
        pair_metrics[pair_name] = metrics
    print(num_valid)
    return pair_metrics
