from scipy.signal import find_peaks, medfilt
from evaluation.structures import BBox, ResponseTrack, BBox3D
import os
import torch
import decord
from dataset import dataset_utils
from evaluation.test_dataloader import load_query, load_clip, process_inputs
from einops import rearrange
from utils import vis_utils, exp_utils
import torchvision
from typing import List, Dict, Any, Tuple
from utils.pcd_utils import get_pcd_in_box_mask, resample_pcd, crop_and_center_pcd, crop_pcd_axis_aligned, Optional
from utils.bounding_box import BoundingBox
from utils.point_cloud import PointCloud
from utils.pcd_utils import get_pcd_in_box_mask, resample_pcd, crop_and_center_pcd, crop_pcd_axis_aligned
from pyquaternion import Quaternion
import numpy as np
import os.path as osp
from PIL import Image
import open3d as o3d
import json
import glob
import torch.nn.functional as F
from torch.cuda.amp import autocast
from utils.mathConvert import (
    compute_3d_box_vertices,
    compute_rotation_matrix_from_directions,
    project_3d_to_2d,
)
SMOOTHING_SIGMA = 5
DISTANCE = 25
WIDTH = 3
PROMINENCE = 0.2
PEAK_SCORE_THRESHILD = 0.5  
PEAK_WINDWOW_RATIO = 0.5

PEAK_SCORE_THRESHOLD = 0.8
PEAK_WINDOW_THRESHOLD = 0.7


class Task:
    def __init__(self, config, annots):
        super().__init__()
        self.config = config
        self.annots = annots
        self.clip_uid = annots[0]["clip_uid"]
        self.annotation_uid = annots[0]["metadata"]["annotation_uid"] # batch1_scene643_clip1_anno1
        self.video_start_sec = annots[0]["metadata"]["video_start_sec"]
        self.video_end_sec = annots[0]["metadata"]["video_end_sec"]
        for annot in self.annots:
            assert annot["clip_uid"] == self.clip_uid
        self.keys = [
            (annot["metadata"]["annotation_uid"], annot["metadata"]["query_set"])
            for annot in self.annots
        ]
        """
        sample: 
        'video_uid' = 'batch1_scene643'
        'video_start_sec' = 0
        'video_end_sec' = 105
        """
        self.batch_id = int(self.annotation_uid.split('_')[0].replace('batch',''))
        self.scene_id = int(self.annotation_uid.split('_')[1].replace('scene',''))
        self.clip_id = int(self.annotation_uid.split('_')[2].replace('clip',''))
        self.data_dir = config.dataset.root
        if self.config.dataset.padding_value == 'zero':
            self.padding_value = 0
        elif self.config.dataset.padding_value == 'mean':
            # 你可以把 NORMALIZE_MEAN 设为 tuple，如果需要就恢复
            self.padding_value = 0.5

    def run(self, config, device=None):
        # 如果没有传 device，就自动选择
        if device is None:
            device = select_device_auto()
        # 针对 4090 做额外优化（如果自动检测到）
        apply_4090_tuning_if_needed(device)

        clip_uid = self.annots[0]["clip_uid"]
        assert clip_uid is not None 

        all_pred_rts = {}
        for key, annot in zip(self.keys, self.annots):
            annotation_uid = annot["metadata"]["annotation_uid"]
            query_set = annot["metadata"]["query_set"]
            annot_key = f"{annotation_uid}_{query_set}"
            query_frame = annot['query_frame']
            visual_crop = annot["visual_crop"]
            save_path = os.path.join(self.config.output_dir,self.config.dataset.name,"infer_outputs/like_ego4d", f'{annot_key}.pt')
            # 检查文件是否存在, 打印警告信息
            assert os.path.isfile(save_path), f"File not found: {save_path}"
            cache = torch.load(save_path)
            ret_bboxes, ret_scores = cache['ret_bboxes'], torch.sigmoid(cache['ret_scores'])
            ret_bboxes = ret_bboxes.numpy()     # bbox in [N,9], original resolution, cv2 axis
            ret_scores = ret_scores.numpy().astype(np.float32)     # scores in [N] to float32

            ret_scores_sm = ret_scores.copy()
            for i in range(1):
                ret_scores_sm = medfilt(ret_scores_sm, kernel_size=SMOOTHING_SIGMA)

            # only used for testing stAP with gt window 
            # gt_scores = np.zeros_like(ret_scores_sm)
            # len_clip = gt_scores.shape[0]
            # gt_rt_idx = [int(frame_it['frame_number']) for frame_it in annot['response_track']]
            # for frame_it in gt_rt_idx:
            #     gt_scores[min(frame_it, len_clip-1)] = random.uniform(0.6,1)
            # ret_scores_sm = gt_scores.copy()

            peaks, _ = find_peaks(ret_scores_sm)
            if len(peaks) == 0:
                print(ret_scores_sm)
            peaks = process_peaks(peaks, ret_scores_sm)

            recent_peak = None
            for peak in peaks[::-1]:
                recent_peak = int(peak)
                break

            if recent_peak is not None:
                threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
                latest_idx = [recent_peak]
                for idx in range(recent_peak, -1, -1):
                    if ret_scores_sm[idx] >= threshold:
                        latest_idx.append(idx)
                    else:
                        break
                for idx in range(recent_peak, query_frame-self.video_start_sec + 1):
                    if ret_scores_sm[idx] >= threshold:
                        latest_idx.append(idx)
                    else:
                        break
            else:
                latest_idx = [query_frame-self.video_start_sec]
            
            latest_idx = sorted(list(set(latest_idx)))
            latest_bbox = ret_bboxes[latest_idx]    # [t,9]
            
            latest_bbox_format = []
            for (frame_bbox, fram_idx) in zip(latest_bbox, latest_idx):
                x, y, z, l, w, h, roll, pitch, yaw = frame_bbox
                bbox_format = BBox3D(fram_idx + self.video_start_sec, x, y, z, w, l, h, roll, pitch, yaw)
                latest_bbox_format.append(bbox_format)
            
            pred_rts = [ResponseTrack(latest_bbox_format, score=1.0)]
            all_pred_rts[key] = pred_rts
        
        return all_pred_rts
    
def process_peaks(peaks_idx, ret_scores_sm):
    '''process the peaks based on their scores'''
    num_frames = ret_scores_sm.shape[0]
    if len(peaks_idx) == 0:
        start_score, end_score = ret_scores_sm[0], ret_scores_sm[-1]
        if start_score > end_score:
            valid_peaks_idx = [0]
        else:
            valid_peaks_idx = [num_frames-1]
    else:
        peaks_score = ret_scores_sm[peaks_idx]
        largest_score = np.max(peaks_score)

        threshold = largest_score * PEAK_SCORE_THRESHOLD

        valid_peaks_idx_idx = np.where(peaks_score > threshold)[0]
        valid_peaks_idx = peaks_idx[valid_peaks_idx_idx]
    return valid_peaks_idx


def select_device_auto():
    """
    自动选择 device。如果有 CUDA 则返回 cuda:0，否则返回 cpu。
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
    
def apply_4090_tuning_if_needed(device):
    """
    针对 RTX 4090 / Ada GPU 做一些全局设置（如果检测到 CUDA 设备）。
    这些设置对其他 CUDA GPU 也通常是有益的，但我们只在 CUDA 可用时执行。
    """
    if device.type != 'cuda':
        return

    try:
        name = torch.cuda.get_device_name(device.index)
    except Exception:
        name = ""

    # 启用 cudnn.benchmark 可以加速固定输入尺寸的网络
    torch.backends.cudnn.benchmark = True

    # 允许 TF32（在不影响数值稳定性的前提下提升矩阵乘法速度）
    # 新版本 torch: torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        # 旧版本 torch 可能没有该属性
        pass

    # 在支持的新 torch 版本上可以调整 float32 matmul 精度策略
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # 这里基于设备名做一个简单判断（如果是 Ada/4090，会包含 '4090' / 'Ada' / 'RTX' 等）
    # 无论是否 4090，以上设置在多数现代 GPU 上都是安全且有益的
    # 你也可以在 config 中添加开关来开启/关闭 channels_last 或 fp16
    return


