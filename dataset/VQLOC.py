import bisect
import glob
import json
import math
import os
import os.path as osp
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm
import torchvision.transforms
from transforms3d.euler import euler2mat
# 项目路径
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# 项目内导入（保持与原逻辑一致）
# from dataset import dataset_utils
from utils.bounding_box import BoundingBox
from utils.point_cloud import PointCloud
from utils.pcd_utils import get_pcd_in_box_mask, resample_pcd, crop_and_center_pcd, crop_pcd_axis_aligned

from .dataset_utils import *
from utils.mathConvert2 import (
    compute_3d_box_vertices,
    compute_rotation_matrix_from_directions,
    project_3d_to_2d,
)
from utils.pl_ddp_rank import pl_ddp_rank
from dataset.base_dataset import QueryVideoDataset

# Matplotlib 后端（仅在需要时使用）
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: F401
from matplotlib.figure import Figure  # noqa: F401

random.seed(0)
K = np.array([[1044.61, 0, 642.33],
              [0, 1046.04, 367.719],
              [0, 0, 1]])
E = np.array([[-0.019445, -0.999799, -0.00484243, 0.0308751],
              [-0.0130265, 0.00509626, -0.999903, -0.326309],
              [0.999617, -0.0263807, -0.0130804, -0.017514],
              [0, 0, 0, 1]])
base_sizes = torch.tensor([
        [0.2402,0.0784,0.0805], # 
        [0.0628,0.1697,0.1582], # 
        [0.5974,0.6246,1.5969],
        [0.0956,0.0846,0.0930],
        [0.1675,0.0650,0.0760],
        [0.1499,0.1200,0.1272],
        [0.2265,0.0515,0.1680],
        [0.0710,0.2297,0.2667],
        [0.2928,0.1390,0.1219],
        [0.1123,0.1038,0.2783],
        [0.2126,0.1648,0.1709],
        [0.1579,0.3081,0.4045],
        [0.1719,0.2143,0.2747],
        [0.4819,0.4835,1.1237],
        [0.3315,0.3258,0.2564],
        [0.3726,0.0786,0.1025],
        [1.8169,0.7726,1.3861],
        [0.3123,0.0843,0.2325],
        [0.3969,0.1257,0.3558],
        [0.4275,0.1938,0.1732],
        [1.2306,0.5972,1.0174],
        [0.3063,0.3426,0.8191],
        [0.1181,0.3690,0.1823],
        [0.6526,0.3509,0.5175],
        [0.9962,0.2993,0.3480],
        [4.5001,1.9692,1.6745],
        [0.7163,0.0875,0.1703],
        [0.2176,0.6172,0.4784],
        [1.4305,0.2895,0.5244],
        [2.7166,1.2378,1.5779],
        [0.4640,1.3243,1.1030],
        [0.5528,0.6575,3.7959]
      ])
aspect_ratios = torch.tensor([1])
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]
n_z = 4

# ------------------------------------------------------------
# 可视化 & 标定相关工具函数（保持原有实现与逻辑）
# ------------------------------------------------------------
def draw_3d_bbox_projection(
    image: np.ndarray,
    bbox_2d_points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制3D框的2D投影线框。
    """
    qs = bbox_2d_points.reshape(-1, 2).astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image


def parse_calib_file(calib_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    解析 JSON 标定文件，返回每个相机的内外参。
    """
    with open(calib_path, 'r') as f:
        camera_data = json.load(f)

    cameras = {}
    for idx, cam in enumerate(camera_data):
        intrinsic = np.array([
            [cam['intrinsic'][0], 0, cam['intrinsic'][2]],
            [0, cam['intrinsic'][1], cam['intrinsic'][3]],
            [0, 0, 1]
        ])
        extrinsic = np.array(cam['extrinsic'])
        cameras[f'camera_{idx}'] = {'intrinsic': intrinsic, 'extrinsic': extrinsic}
    return cameras


def are_points_in_box(
    points: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
    rotation: np.ndarray
) -> np.ndarray:
    """
    判断点是否在3D框内。
    """
    order = 'rxyz'
    r = euler2mat(rotation[0], rotation[1], rotation[2], order)
    local_points = (points - center) @ r
    half_dim = np.array(dimensions) / 2.0
    inside_mask = np.all(np.abs(local_points) <= half_dim, axis=1)
    return inside_mask


def create_lineset_from_box(vertices: np.ndarray) -> o3d.geometry.LineSet:
    """
    用 8 个顶点创建 LineSet。
    """
    lines = [
        [0, 1], [1, 5], [5, 4], [4, 7],
        [3, 2], [2, 6], [6, 7], [7, 3],
        [0, 3], [1, 2], [5, 6], [4, 7]
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_arrow(origin: np.ndarray, rot_matrix: np.ndarray, frame_info: Dict[str, Any], color=(1, 0, 0)):
    """创建一个指向给定方向的箭头。"""
    length = frame_info["size3D"][0] / 2.0
    radius = min(frame_info["size3D"][0], frame_info["size3D"][1]) / 10.0
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=1.5 * radius,
        cone_height=4.0 * radius,
        cylinder_radius=radius,
        cylinder_height=length
    )
    mesh_arrow.paint_uniform_color(color)
    default_direction = np.array([0, 0, 1])
    target_direction = np.array([1, 0, 0])
    r_default_to_target = compute_rotation_matrix_from_directions(default_direction, target_direction)
    mesh_arrow.rotate(rot_matrix @ r_default_to_target, center=(0, 0, 0))
    mesh_arrow.translate(origin)
    return mesh_arrow


def prepare_vis_lists(img: Image.Image, calib_file: str, frame_info: Dict[str, Any], point_size: float = 0.5) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    通过投影3D框计算2D框，不修改传入的 frame_info。
    始终返回三元组 (bbox_dict, intrinsic, extrinsic)。失败时 bbox 返回全0，内外参返回单位阵。
    """
    def to_vec3(v):
        if v is None:
            return None
        if isinstance(v, dict):
            return np.array([v.get('x', 0.0), v.get('y', 0.0), v.get('z', 0.0)], dtype=float)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size < 3:
            b = np.zeros(3, dtype=float)
            b[:a.size] = a
            return b
        return a[:3]

    rot = to_vec3(frame_info.get('rotation3D') if isinstance(frame_info, dict) else None)
    cen = to_vec3(frame_info.get('center3D') if isinstance(frame_info, dict) else None)
    siz = to_vec3(frame_info.get('size3D') if isinstance(frame_info, dict) else None)

    if rot is None or cen is None or siz is None:
        return (
            {'x_min': 0.0, 'y_min': 0.0, 'x_max': 0.0, 'y_max': 0.0},
            np.eye(3, dtype=float),
            np.eye(4, dtype=float),
        )

    img = np.array(img)
    box_vertices = compute_3d_box_vertices(cen, siz, rot)
    calib_infos = parse_calib_file(calib_file)
    intrinsic_camera_0 = calib_infos['camera_0']['intrinsic']
    extrinsic_camera_0 = calib_infos['camera_0']['extrinsic']
    box_vertices_2d_camera_0 = None if box_vertices is None else project_3d_to_2d(
        box_vertices, intrinsic_camera_0, extrinsic_camera_0
    )
    if box_vertices_2d_camera_0 is not None:
        try:
            box_vertices_2d_camera_0 = np.asarray(box_vertices_2d_camera_0, dtype=float)
        except Exception:
            # 兜底转换为空
            box_vertices_2d_camera_0 = None
    if box_vertices_2d_camera_0 is not None:
        try:
            box_vertices_2d_camera_0 = np.asarray(box_vertices_2d_camera_0, dtype=float)
        except Exception:
            # 兜底转换为空
            box_vertices_2d_camera_0 = None
    if box_vertices_2d_camera_0 is None or len(box_vertices_2d_camera_0) == 0:
        return (
            {'x_min': 0.0, 'y_min': 0.0, 'x_max': 0.0, 'y_max': 0.0},
            intrinsic_camera_0,
            extrinsic_camera_0,
        )

    return (
        {
            'x_min': float(np.min(box_vertices_2d_camera_0[:, 0])),
            'y_min': float(np.min(box_vertices_2d_camera_0[:, 1])),
            'x_max': float(np.max(box_vertices_2d_camera_0[:, 0])),
            'y_max': float(np.max(box_vertices_2d_camera_0[:, 1])),
        },
        intrinsic_camera_0,
        extrinsic_camera_0,
    )


def visualize_projection(image: Image.Image, img_bbox: Dict[str, float], save_path: str = 'aaaaaaa.jpg', show: bool = False):
    """
    可视化3D到2D投影结果。
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        if not show and save_path:
            matplotlib.use('Agg')

        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)

        rect = plt.Rectangle(
            (img_bbox['x_min'], img_bbox['y_min']),
            img_bbox['x_max'] - img_bbox['x_min'],
            img_bbox['y_max'] - img_bbox['y_min'],
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(img_bbox['x_min'], img_bbox['y_min'] - 10,
                f"({img_bbox['x_min']:.1f}, {img_bbox['y_min']:.1f})",
                color='red', fontsize=12, weight='bold')
        ax.text(img_bbox['x_max'], img_bbox['y_max'] + 20,
                f"({img_bbox['x_max']:.1f}, {img_bbox['y_max']:.1f})",
                color='red', fontsize=12, weight='bold')
        ax.set_title('3D BBox Projection to 2D Image', fontsize=14, weight='bold')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

    except Exception as e:
        print(f"可视化出错: {e}")
        print(f"边界框信息: {img_bbox}")
        print(f"图像大小: {image.size}")


# ------------------------------------------------------------
# 数据集类
# ------------------------------------------------------------
class VQLOC(QueryVideoDataset):
    def __init__(self, dataset_name, query_params, clip_params, data_dir, split: str = 'train'):
        super().__init__(dataset_name, query_params, clip_params, data_dir, split=split)
        self.query_params = query_params
        self.clip_params = clip_params
        self.split = split
        self.total_samples = 0

        all_batch_ids = self._get_available_batch_ids()
        self.train_ratio = 1.0
        # all_batch_ids = [9]
        # all_batch_ids = [1, 2, 3]
        self.annotations = self._build_annotations(all_batch_ids)
        random.shuffle(self.annotations)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 仅用标注采样，不预读所有帧，降低CPU/IO
        search_tracklet_anno = self.annotations[idx]['search']
        template_frame_anno = self.annotations[idx]['template']

        # 基于标注得到有效范围 -> 采样帧索引
        sample = self._get_valid_ranges_from_annos(search_tracklet_anno)
        frame_idxs = self.sample_frames_balance(
            self.clip_params['clip_num_frames'],
            self.clip_params['frame_interval'],
            sample
        )

        # 按需读取被采样的帧
        search_frames = [self._build_search_frame(search_tracklet_anno[i]) for i in frame_idxs]
        template_frame = self._build_template_frame(template_frame_anno)

        return self._generate_item(template_frame, search_frames)

    # ---------------------
    # 文件/路径/扫描工具
    # ---------------------
    @staticmethod
    def _find_matching_json_file(pattern_path: str) -> Optional[str]:
        """
        使用通配符模式查找匹配的JSON文件。
        """
        try:
            matching_files = glob.glob(pattern_path)
            if matching_files:
                return matching_files[0]
            else:
                dir_path = osp.dirname(pattern_path)
                if osp.exists(dir_path):
                    available_files = os.listdir(dir_path)
                    print(f"No matching files found for pattern: {pattern_path}")
                    print(f"Available files in directory {dir_path}: {available_files}")
                else:
                    print(f"Directory does not exist: {dir_path}")
                return None
        except Exception as e:
            print(f"Error finding matching file for pattern {pattern_path}: {e}")
            return None

    def _get_pcd_folder_path(self, batch_id: int, scene_id: int, data_type: str = 'o_s') -> str:
        """
        自动选择合适的pcd文件夹路径，优先 lidar_point_cloud_0，其次 point_cloud_bin。
        """
        base_path = osp.join(self.data_dir, f"batch{batch_id}", 'img', data_type, 'Seq_%06d' % scene_id)
        lidar_folder = osp.join(base_path, 'lidar_point_cloud_0')
        point_cloud_bin_folder = osp.join(base_path, 'point_cloud_bin')

        if osp.exists(lidar_folder):
            files = os.listdir(lidar_folder)
            pcd_files = [f for f in files if f.endswith('.pcd')]
            if len(pcd_files) > 0:
                return 'lidar_point_cloud_0'

        if osp.exists(point_cloud_bin_folder):
            files = os.listdir(point_cloud_bin_folder)
            pcd_files = [f for f in files if f.endswith('.pcd')]
            if len(pcd_files) > 0:
                return 'point_cloud_bin'
        return 'lidar_point_cloud_0'

    def _get_available_batch_ids(self) -> List[int]:
        """
        扫描数据目录，获取所有可用的批次ID。
        """
        batch_ids: List[int] = []
        search_dir = osp.join(self.data_dir)
        if osp.exists(search_dir):
            for folder_name in os.listdir(search_dir):
                if folder_name.startswith('batch') and osp.isdir(osp.join(search_dir, folder_name)):
                    batch_id = int(folder_name[5:])
                    batch_ids.append(batch_id)
        batch_ids.sort()
        return batch_ids

    def _get_available_scene_ids(self, batch_id: int) -> List[int]:
        """
        扫描数据目录，获取所有可用的场景ID。
        """
        scene_ids: List[int] = []
        search_dir = osp.join(self.data_dir, f"batch{batch_id}", 'lable', 'o_s')
        if osp.exists(search_dir):
            for folder_name in os.listdir(search_dir):
                if folder_name.startswith('Seq_') and osp.isdir(osp.join(search_dir, folder_name)):
                    scene_id = int(folder_name.split('_')[1])
                    scene_ids.append(scene_id)
        scene_ids.sort()
        return scene_ids

    # ---------------------
    # 采样/预处理
    # ---------------------
    def filter_tracklet(self, t_annos, ts, min_tracklet_length: int = 0):
        t_annos_new, ts_new = [], []
        for t_anno, t in zip(t_annos, ts):
            if len(t_anno) >= min_tracklet_length:
                t_annos_new.append(t_anno)
                ts_new.append(t)
        return t_annos_new, ts_new

    def _get_valid_ranges_from_annos(self, search_tracklet_anno: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于标注中 contour 是否为 None 来得到多段有效范围与总帧数。仅读取标注。
        """
        valid_frames = [i for i, a in enumerate(search_tracklet_anno) if a.get('contour') is not None]
        valid_ranges = []
        if valid_frames:
            start = prev = valid_frames[0]
            for fid in valid_frames[1:]:
                if fid != prev + 1:
                    valid_ranges.append([start, prev])
                    start = fid
                prev = fid
            valid_ranges.append([start, prev])
        return {
            "response_track_valid_ranges": valid_ranges,
            "total_frames": len(search_tracklet_anno),
        }

    def sample_frames_balance(self, num_frames: int, frame_interval: int, sample: Dict[str, Any], sampling: str = 'rand') -> List[int]:
        """
        从严格网格 grid=range(0,total_frames,frame_interval) 中，选取一个连续窗口（长度=num_frames），
        使窗口内正负样本尽量 1:1；相邻帧索引恒为 frame_interval；最终升序返回。
        若全局任何窗口都无法达到 1:1，则选择最接近的窗口。
        """
        valid_ranges = sample.get("response_track_valid_ranges", [])
        total_frames = int(sample.get("total_frames", 0))
        frame_interval = max(1, int(frame_interval))
        num_frames = int(max(0, num_frames))

        if total_frames <= 0 or num_frames == 0:
            return []

        # 严格间隔的全局网格（保证相邻差恒为 frame_interval）
        grid = list(range(0, total_frames, frame_interval))
        if len(grid) == 0:
            return []

        # 若网格长度不足，则缩短期望长度（不能保持等间隔的情况下不做重复填充）
        win_len = min(num_frames, len(grid))

        def in_valid(i: int) -> bool:
            for s, e in valid_ranges:
                if s <= i <= e:
                    return True
            return False

        # 预计算每个 grid 索引是否为正样本
        pos_mask = [in_valid(i) for i in grid]

        # 滑动窗口，挑选最平衡的窗口
        best_start, best_cost, best_pos_cnt = 0, float('inf'), -1
        best_start_list = []
        for start in range(0, len(grid) - win_len + 1):
            end = start + win_len
            pos_cnt = sum(pos_mask[start:end])
            # 次级准则：在 cost 相同的情况下，优先 pos_cnt 更接近 win_len/2 的（其实已由 cost 体现）
            # 再次：可偏好更多非空（至少包含正/负各一个）
            if (pos_cnt >= best_cost):
                best_cost = pos_cnt 
                best_start = start
            if pos_cnt == 10: # and pos_cnt == target_pos:
                best_start_list.append(start)
        if len(best_start_list) != 0:
            best_start = random.choice(best_start_list)  # 多个最优解时随机选
        selected = grid[best_start: best_start + win_len]
        # 已保证升序与恒定间隔
        return selected

    def process_pcd_bbox(self, pcd_bbox) -> torch.Tensor:
        """
        将 BoundingBox 转为 [cx, cy, cz, l, w, h, roll, pitch, yaw]。
        """
        center = pcd_bbox.center
        wlh = pcd_bbox.wlh
        rot = pcd_bbox.rot
        # 设置最小尺寸阈值，避免过小
        min_size = 0.02
        wlh = [max(dim, min_size) for dim in wlh]
        # norm [0,1]
        # center[0] = (center[0]-space_range[0][0]) / (space_range[1][0] - space_range[0][0])
        # center[1] = (center[1]-space_range[0][1]) / (space_range[1][1] - space_range[0][1])
        # center[2] = (center[2]-space_range[0][2]) / (space_range[1][2] - space_range[0][2])
        # rot[0] = rot[0] / (2 * math.pi)  # roll 归一化到 [-0.5,0.5]
        # rot[1] = rot[1] / (2 * math.pi)  # pitch 归一化到 [-0.5,0.5]
        # rot[2] = rot[2] / (2 * math.pi)  # yaw 归一化到 [-0.5,0.5]
         # cx, cy, cz, l, w, h, roll, pitch, yaw
        return torch.tensor([center[0], center[1], center[2], wlh[1], wlh[0], wlh[2], rot[0], rot[1], rot[2]])

    def process_pcd(self, pcd, bbox) -> torch.Tensor:
        """
        重采样点云，返回 (N, 3)，与 process_bbox_pcd 保持一致（末维为 xyz）。
        """
        # pcd_crop = crop_pcd_axis_aligned(
        #         pcd, bbox
        #     )
        # pcd_rs, _ = resample_pcd(pcd, self.clip_params['frame_npts'], return_idx=True, is_training=True)
        pcd_rs, _ = resample_pcd(pcd, 4096, return_idx=True, is_training=True)
        pts3 = pcd_rs.points[:3, :].T  # (N, 3)
        return torch.as_tensor(pts3, dtype=torch.float32)

    def get_valid_points(self, pcd):
        """
        获取有效点的掩码。
        投影到2d上在图片上的点
        Args: 
            pcd : np.ndarray (N,3)
        """
        h = 720
        w = 1280
        pcd_2d = project_3d_to_2d(
            pcd, K, E
        )  # (N,2)
        pcd_2d = np.asarray(pcd_2d, dtype=float)
        valid_mask = (
            (pcd_2d[..., 0] >= 0) & 
            (pcd_2d[..., 0] < w) & 
            (pcd_2d[..., 1] >= 0) & 
            (pcd_2d[..., 1] < h) &
            (pcd[..., 0] > 0)  # 前方
        )
        # pcd_2d = pcd_2d[valid_mask]
        pts = pcd[valid_mask]
        return pts

        return pcd

    def recover_bbox(self, bbox: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """
        将 [0,1] 归一化坐标的 bbox 还原到像素坐标系。
        """
        bbox_cp = bbox.clone()
        if len(bbox.shape) > 1:
            bbox_cp[..., 0] *= img_w
            bbox_cp[..., 1] *= img_h
            bbox_cp[..., 2] *= img_w
            bbox_cp[..., 3] *= img_h
            return bbox_cp
        else:
            return torch.tensor([bbox_cp[0] * img_w, bbox_cp[1] * img_h, bbox_cp[2] * img_w, bbox_cp[3] * img_h])

    def normalize_bbox(self, bbox: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """
        将像素坐标系 bbox 归一化到 [0,1]。
        """
        bbox_cp = bbox.clone()
        if len(bbox.shape) > 1:
            bbox_cp[..., 0] /= img_w
            bbox_cp[..., 1] /= img_h
            bbox_cp[..., 2] /= img_w
            bbox_cp[..., 3] /= img_h
            return bbox_cp
        else:
            return torch.tensor([bbox_cp[0] / img_w, bbox_cp[1] / img_h, bbox_cp[2] / img_w, bbox_cp[3] / img_h])

    def create_square_bbox(self, bbox: List[float], img_h: int, img_w: int) -> torch.Tensor:
        """
        将 bbox 扩展为正方形（保持中心不变）。
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = center_x - x1
        h = center_y - y1
        r = max(h, w)

        new_x1 = max(center_x - r, 0)
        new_x2 = min(center_x + r, img_w - 1)
        new_y1 = max(center_y - r, 0)
        new_y2 = min(center_y + r, img_h - 1)
        new_bbox = torch.tensor([new_x1, new_y1, new_x2, new_y2])
        return new_bbox

    # ---------------------
    # 核心数据生成
    # ---------------------
    def change_xyxy_to_yxyx(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        将 bbox 从 [x1,y1,x2,y2] 转为 [y1,x1,y2,x2]。
        """
        if len(bbox.shape) > 1:
            return bbox[:, [1, 0, 3, 2]]
        else:
            return torch.tensor([bbox[1], bbox[0], bbox[3], bbox[2]])

    def _process_clip(self, clip: torch.Tensor, clip_img_bbox: torch.Tensor, clip_with_bbox: torch.Tensor):
        """
        对抽取的视频帧与其 bbox 做 pad/resize，并尝试抽取 query_img。
        """
        target_size = self.clip_params['fine_size']

        t, _, h, w = clip.shape
        clip_img_bbox = self.recover_bbox(clip_img_bbox, h, w)

        max_size, min_size = max(h, w), min(h, w)
        pad_height = True if h < w else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2
            clip_img_bbox[:, 1] += (max_size - min_size) / 2.0
            clip_img_bbox[:, 3] += (max_size - min_size) / 2.0
        else:
            pad_input = [pad_size, 0] * 2
            clip_img_bbox[:, 0] += (max_size - min_size) / 2.0
            clip_img_bbox[:, 2] += (max_size - min_size) / 2.0
        transform_pad = torchvision.transforms.Pad(pad_input, fill=self.padding_value)
        clip = transform_pad(clip)
        # clip = F.pad(clip, pad=tuple(pad_input), mode='constant', value=float(self.padding_value))
        clip = F.interpolate(clip, size=(target_size, target_size), mode='bilinear')
        clip_img_bbox = clip_img_bbox / float(max_size)
        clip_h, clip_w = target_size, target_size
        return clip, clip_img_bbox, clip_with_bbox, clip_h, clip_w

    def _get_query_(self, comp_template: Dict[str, Any]):
        """
        从模板帧生成 query_img 和其归一化 bbox。
        额外返回模板帧原始尺寸 (orig_h, orig_w)，用于相机投影映射。
        """
        target_size = self.clip_params['fine_size']
        query_img = comp_template['img']                 # PIL
        bbox_dict = comp_template['img_bbox']
        bbox = torch.tensor([
            bbox_dict.get('x_min', 0.0),
            bbox_dict.get('y_min', 0.0),
            bbox_dict.get('x_max', 0.0),
            bbox_dict.get('y_max', 0.0),
        ], dtype=torch.float32)

        w0, h0 = query_img.size  # PIL: (W,H) 原始尺寸
        if self.query_params.get('query_square', False):
            bbox = self.create_square_bbox(bbox.tolist(), h0, w0)

        # 对称 pad 到正方形，再 resize 到 target_size
        max_size, min_size = max(h0, w0), min(h0, w0)
        pad_height = True if h0 < w0 else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2
            bbox[1] = bbox[1] + (max_size - min_size) / 2.0
            bbox[3] = bbox[3] + (max_size - min_size) / 2.0
        else:
            pad_input = [pad_size, 0] * 2
            bbox[0] = bbox[0] + (max_size - min_size) / 2.0
            bbox[2] = bbox[2] + (max_size - min_size) / 2.0
        transform_pad = torchvision.transforms.Pad(pad_input, fill=self.padding_value)
        query_img_pad = transform_pad(query_img)
        query_img_pad = query_img_pad.resize((target_size, target_size))
        query_img_t = torch.from_numpy(np.asarray(query_img_pad) / 255.0).permute(2, 0, 1).contiguous().clone()

        bbox = bbox / float(max_size)  # 归一化到 pad 后的 max_size
        return query_img_t, bbox, h0, w0

    def _generate_item(self, comp_template: Dict[str, Any], search_frame: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        生成一次训练用的样本字典。
        """
        _w, _h = comp_template['img'].size  # 仅用于保持与原逻辑一致（可能未使用）

        # 图像与点云
        img_transform = torchvision.transforms.ToTensor()
        clip_img = torch.stack([img_transform(frame['img']) for frame in search_frame])
        # 生成点云序列 (T, N, 3)
        clip_pcd = torch.stack([self.process_pcd(frame['pcd'], frame['pcd_bbox']) for frame in search_frame])

        # 每帧各自尺寸归一化 bbox，且按键名顺序取值
        clip_img_bbox_px: List[torch.Tensor] = []
        per_frame_hw: List[Tuple[int, int]] = []  # (H,W)
        for frame in search_frame:
            bbox_d = frame['img_bbox']
            clip_img_bbox_px.append(torch.tensor([
                bbox_d.get('x_min', 0.0), bbox_d.get('y_min', 0.0),
                bbox_d.get('x_max', 0.0), bbox_d.get('y_max', 0.0),
            ], dtype=torch.float32))
            w_i, h_i = frame['img'].size  # PIL size = (W,H)
            per_frame_hw.append((h_i, w_i))
        clip_img_bbox_px = torch.stack(clip_img_bbox_px)
        clip_img_bbox = torch.stack([
            self.normalize_bbox(clip_img_bbox_px[i], per_frame_hw[i][0], per_frame_hw[i][1])
            for i in range(len(search_frame))
        ])

        clip_pcd_bbox = torch.stack([self.process_pcd_bbox(frame['pcd_bbox']) for frame in search_frame])
        clip_with_bbox = torch.stack([
            torch.tensor(True, dtype=torch.bool) if frame['anno']['contour'] is not None else torch.tensor(False, dtype=torch.bool)
            for frame in search_frame
        ])

        # 将图像序列 pad->resize 到方形，同时归一化 bbox
        clip_img, clip_img_bbox, clip_with_bbox, clip_h, clip_w = self._process_clip(
            clip_img, clip_img_bbox, clip_with_bbox
        )
        # 模板（query_frame）对应的图像/点云
        query_frame_img, query_frame_img_bbox, q_h0, q_w0 = self._get_query_(comp_template)
        query_frame_pcd = self.process_pcd(comp_template['pcd'], comp_template['pcd_bbox'])  # (N,3)
        query_frame_pcd_bbox = self.process_pcd_bbox(comp_template['pcd_bbox']) # (9,)

        clip_img_bbox = self.change_xyxy_to_yxyx(clip_img_bbox)  # [T,4]
        query_frame_img_bbox = self.change_xyxy_to_yxyx(query_frame_img_bbox)  # [4]

        data = {
            # 'clip_img': clip_img.float(),
            # 'clip_pcd': clip_pcd.float(),
            # 'clip_with_bbox': clip_with_bbox.float(),
            # 'clip_img_bbox': clip_img_bbox.float().clamp(min=0.0, max=1.0),
            # 'clip_pcd_bbox': clip_pcd_bbox.float(),
            # 'clip_h': clip_h,
            # 'clip_w': clip_w,
            # 'query_frame_img': query_frame_img.float(),
            # 'query_frame_img_bbox': query_frame_img_bbox.float(),
            # 'query_frame_pcd': query_frame_pcd.float(),
            # 'query_frame_pcd_bbox': query_frame_pcd_bbox.float(),
            # 'cam' : {
            #     'intrinsic': comp_template['intrinsic'],
            #     'extrinsic': comp_template['extrinsic'],
            #     'clip': {'img_h': clip_h, 'img_w': clip_w},         # 搜索帧原始尺寸
            #     'query': {'img_h': q_h0, 'img_w': q_w0},            # 模板帧原始尺寸
            # },
            
            'clip_with_bbox': clip_with_bbox.float(),       # [T]
            'before_query': torch.ones_like(clip_with_bbox).bool(), #before_query.bool(),

            'clip' : clip_img.float(),                           # [T,3,H,W]
            'clip_bbox': clip_img_bbox.float().clamp(min=0.0, max=1.0),                 # [T,4]
            'query': query_frame_img.float(),                         # [3,H2,W2]
            'clip_h': torch.tensor(clip_h),
            'clip_w': torch.tensor(clip_w),
            'query_frame': query_frame_img.float(),             # [3,H,W]
            'query_frame_bbox': query_frame_img_bbox.float(),    # [4]

            'clip_pcd': clip_pcd.float(),                         # [T,N,3]
            'clip_pcd_bbox' : clip_pcd_bbox.float(),        # [T, 9]
            'query_frame_pcd': query_frame_pcd.float(),         # [N,3]
            'query_frame_pcd_bbox': query_frame_pcd_bbox.float(), # [9]
            'cam' : {
                'intrinsic': comp_template['intrinsic'],
                'extrinsic': comp_template['extrinsic'],
                'clip': {'img_h': clip_h, 'img_w': clip_w},        # 搜索帧原始尺寸
                'query': {'img_h': q_h0, 'img_w': q_w0},           # 模板帧原始尺寸
            }
        }


        return data
    # results = {
    #         'clip': clip.float(),                           # [T,3,H,W]
    #         'clip_with_bbox': clip_with_bbox.float(),       # [T]
    #         'before_query': torch.ones_like(clip_with_bbox).bool(), #before_query.bool(),            # [T]
    #         'clip_bbox': clip_bbox.float().clamp(min=0.0, max=1.0),                 # [T,4]
    #         'query': query.float(),                         # [3,H2,W2]
    #         'clip_h': torch.tensor(clip_h),
    #         'clip_w': torch.tensor(clip_w),
    #         'query_frame': query_frame.float(),             # [3,H,W]
    #         'query_frame_bbox': query_frame_bbox.float()    # [4]
    #     }
    # ---------------------
    # I/O 构建帧/标注
    # ---------------------
    def _build_annotations(self, batch_ids: List[int]) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        for batch_id in batch_ids:
            scene_ids = self._get_available_scene_ids(batch_id)
            if self.split == 'train':
                scene_ids = scene_ids[:int(len(scene_ids) * self.train_ratio)]
            else:
                scene_ids = scene_ids[int(len(scene_ids) * self.train_ratio):]
            random.shuffle(scene_ids)
            for scene_id in tqdm(scene_ids, desc='[%6s]Loading annos' % self.split.upper(), disable=pl_ddp_rank() != 0):
                search_file_dir = osp.join(
                    self.data_dir, f"batch{batch_id}", 'lable', 'o_s',
                    'Seq_%06d' % scene_id, 'Seq_%06d.json' % scene_id
                )
                template_file_dir = osp.join(
                    self.data_dir, f"batch{batch_id}", 'lable', 'o_s_t',
                    'Seq_%06d' % scene_id, 'Seq_%06d.json' % scene_id
                )
                search_data = pd.read_json(search_file_dir)
                template_data = pd.read_json(template_file_dir)
                search_objects_series = search_data['objects']
                template_objects_series = template_data['objects']

                search_objects = []
                template_objects = []
                for obj_list, i in zip(search_objects_series, range(len(search_objects_series))):
                    if len(obj_list) > 0 and isinstance(obj_list, list):
                        obj_list[0]['scene'] = scene_id
                        obj_list[0]['frame'] = i + 1
                        obj_list[0]['batch_id'] = batch_id
                        search_objects.extend(obj_list)
                    else:
                        none_objects = {'id': None, 'type': None, 'classId': None, 'className': None, 'trackId': None, 'trackName': None, 'classValues': None, 'contour': None, 'modelConfidence': None, 'modelClass': '', 'scene': scene_id, 'frame': i + 1, 'batch_id': batch_id}
                        search_objects.append(none_objects)

                for obj_list, i in zip(template_objects_series, range(len(template_objects_series))):
                    if len(obj_list) > 0 and isinstance(obj_list, list):
                        obj_list[0]['scene'] = scene_id
                        obj_list[0]['frame'] = i + 1
                        obj_list[0]['batch_id'] = batch_id
                        template_objects.extend(obj_list)

                if len(template_objects) > 1:
                    pcd_folder = self._get_pcd_folder_path(batch_id, scene_id, 'o_s_t')
                    tem_file_dir = osp.join(
                        self.data_dir, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, pcd_folder)
                    file_names = os.listdir(tem_file_dir)[0]
                    frame_id = int(file_names.split('.')[0])
                    template_objects = [item for item in template_objects if item.get("frame") == frame_id]

                if not search_objects or len(search_objects) == 0:
                    continue
                if not template_objects or len(template_objects) == 0:
                    continue

                search_df = pd.DataFrame(search_objects)
                template_df = pd.DataFrame(template_objects)
                search_tracklet_anno = [frame_anno.to_dict() for _, frame_anno in search_df.iterrows()]
                template_tracklet_anno = [frame_anno.to_dict() for _, frame_anno in template_df.iterrows()]

                annotations.append({'batch_id': batch_id, 'scene_id': scene_id, 'search': search_tracklet_anno, 'template': template_tracklet_anno[0]})
        return annotations

    def _read_calibration_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        读取 KITTI 风格标定文件（未在当前流程中使用，保留原逻辑）。
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                try:
                    data[values[0]] = np.array([float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data
    
    def _get_calib_file_path(self, batch_id: int, scene_id: int, data_type: str) -> Optional[str]:
        """
        缓存每个 scene 的标定文件路径，避免每帧反复 glob。
        """
        if not hasattr(self, "_calib_cache"):
            self._calib_cache = {}
        key = (batch_id, scene_id, data_type)
        if key in self._calib_cache:
            return self._calib_cache[key]
        pattern = osp.join(self.data_dir, f"batch{batch_id}", 'img', data_type, f'Seq_{scene_id:06d}', 'Seq_*.json')
        path = self._find_matching_json_file(pattern)
        self._calib_cache[key] = path
        return path
    
    def _build_search_frame(self, frame_anno: Dict[str, Any], bbox_only: bool = False) -> Dict[str, Any]:
        scene_id = frame_anno['scene']
        frame_id = frame_anno['frame']
        batch_id = int(frame_anno['batch_id'])
        frame_info = frame_anno['contour']

        pcd_folder = self._get_pcd_folder_path(batch_id, scene_id, 'o_s')
        pcd_file_dir = osp.join(
            self.data_dir, f"batch{batch_id}", 'img', 'o_s', 'Seq_%06d' % scene_id, f'{pcd_folder}/%05d.pcd' % frame_id)

        pcd_o3d = o3d.io.read_point_cloud(pcd_file_dir)
        pts = np.asarray(pcd_o3d.points, dtype=np.float32) # (N,3) (x,y,z)
        pts = self.get_valid_points(pts)
        if pts.size == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pcd = PointCloud(np.hstack([pts, intensity]).T)

        rgb_file_dir = osp.join(
            self.data_dir, f"batch{batch_id}", 'img', 'o_s', 'Seq_%06d' % scene_id, 'camera_image_0/%05d.jpg' % frame_id)
        rgb = Image.open(rgb_file_dir).convert('RGB')

        camera_file_dir = self._get_calib_file_path(batch_id, scene_id, 'o_s')
        if camera_file_dir is None:
            rgb_bbox = {'x_min': 0, 'y_min': 0, 'x_max': 1e-5, 'y_max': 1e-5}
            pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
            lable = torch.tensor([0], dtype=torch.int64)
            intrinsic = np.eye(3, dtype=float)
            extrinsic = np.eye(4, dtype=float)
        else:
            if frame_anno['contour'] is None:
                rgb_bbox = {'x_min': 0, 'y_min': 0, 'x_max': 1e-5, 'y_max': 1e-5}
                pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
                lable = torch.tensor([0], dtype=torch.int64)
                intrinsic = np.eye(3, dtype=float)
                extrinsic = np.eye(4, dtype=float)
            else:
                rgb_bbox, intrinsic, extrinsic = prepare_vis_lists(rgb, camera_file_dir, frame_info)
                pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
                lable = torch.tensor([1], dtype=torch.int64)
                if rgb_bbox['x_min'] == rgb_bbox['x_max'] or rgb_bbox['y_min'] == rgb_bbox['y_max']:
                    visualize_projection(rgb, rgb_bbox)

        return {'pcd': pcd, 'img': rgb, 'pcd_bbox': pcd_bbox, 'img_bbox': rgb_bbox, 'anno': frame_anno, 'lable': lable, 'intrinsic': intrinsic, 'extrinsic': extrinsic}

    def _build_template_frame(self, template_tracklet_annotation: Dict[str, Any]) -> Dict[str, Any]:
        template_tracklet_annotation = [template_tracklet_annotation]
        scene_id = template_tracklet_annotation[0]['scene']
        batch_id = int(template_tracklet_annotation[0]['batch_id'])

        if len(template_tracklet_annotation) != 1:
            tem_file_dir = osp.join(
                self.data_dir, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, 'lidar_point_cloud_0')
            file_names = os.listdir(tem_file_dir)[0]
            frame_id = int(file_names.split('.')[0])
        else:
            frame_id = template_tracklet_annotation[0]['frame']

        frame_anno = [d for d in template_tracklet_annotation if d.get('frame') == frame_id][0]
        frame_info = frame_anno['contour']

        pcd_folder = self._get_pcd_folder_path(batch_id, scene_id, 'o_s_t')
        pcd_file_dir = osp.join(
            self.data_dir, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, f'{pcd_folder}/%05d.pcd' % frame_id)

        pcd_o3d = o3d.io.read_point_cloud(pcd_file_dir)
        pts = np.asarray(pcd_o3d.points, dtype=np.float32)
        pts = self.get_valid_points(pts)
        if pts.size == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pcd = PointCloud(np.hstack([pts, intensity]).T)

        rgb_file_dir = osp.join(self.data_dir, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, 'camera_image_0/%05d.jpg' % frame_id)
        rgb = Image.open(rgb_file_dir).convert('RGB')

        camera_file_dir = self._get_calib_file_path(batch_id, scene_id, 'o_s_t')
        if camera_file_dir is None:
            rgb_bbox = {'x_min': 0, 'y_min': 0, 'x_max': rgb.width, 'y_max': rgb.height}
            pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
            intrinsic = np.eye(3, dtype=float)
            extrinsic = np.eye(4, dtype=float)
        else:
            rgb_bbox, intrinsic, extrinsic = prepare_vis_lists(rgb, camera_file_dir, frame_info)
            pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
            # if rgb_bbox['x_min'] == rgb_bbox['x_max'] or rgb_bbox['y_min'] == rgb_bbox['y_max']:
            #     visualize_projection(rgb, rgb_bbox)

        return {'pcd': pcd, 'img': rgb, 'pcd_bbox': pcd_bbox, 'img_bbox': rgb_bbox, 'anno': frame_anno, 'intrinsic': intrinsic, 'extrinsic': extrinsic}

    # ---------------------
    # 其他工具
    # ---------------------
    def _to_float_tensor(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tensor_data: Dict[str, torch.Tensor] = {}
        for k, v in data.items():
            if isinstance(v, Image.Image):
                tensor_data[k] = torchvision.transforms.ToTensor()(v)
            elif isinstance(v, dict):
                tensor_data[k] = torch.FloatTensor([vv for vv in v.values()])
            elif isinstance(v, (list, tuple)):
                tensor_data[k] = torch.FloatTensor(v)
            elif isinstance(v, np.ndarray):
                tensor_data[k] = torch.FloatTensor(v)
            elif isinstance(v, bool):
                tensor_data[k] = torch.BoolTensor([v])
            elif isinstance(v, torch.Tensor):
                tensor_data[k] = v
            elif v is None:
                tensor_data[k] = torch.zeros(0, dtype=torch.float32)
            else:
                tensor_data[k] = torch.FloatTensor([v]) if np.isscalar(v) else torch.FloatTensor(v)
        return tensor_data

    @staticmethod
    def _bbox_from_contour(frame_info: Optional[Dict[str, Any]]):
        """
        由 contour 生成 BoundingBox（保持原逻辑）。
        """
        def to_vec3(v, default=(0., 0., 0.)):
            if v is None:
                return np.array(default, dtype=np.float32)
            if isinstance(v, dict):
                return np.array([v.get('x', 0.), v.get('y', 0.), v.get('z', 0.)], dtype=np.float32)
            a = np.asarray(v, dtype=np.float32).reshape(-1)
            if a.size < 3:
                b = np.zeros(3, dtype=np.float32)
                b[:a.size] = a
                return b
            return a[:3]

        cen = to_vec3(frame_info.get('center3D') if isinstance(frame_info, dict) else None)
        siz = to_vec3(frame_info.get('size3D') if isinstance(frame_info, dict) else None)
        rot = to_vec3(frame_info.get('rotation3D') if isinstance(frame_info, dict) else None)

        L, W, H = float(siz[0]), float(siz[1]), float(siz[2])
        size_wlh = [W, L, H]

        yaw = float(rot[2])
        orientation = Quaternion(axis=[0, 0, 1], radians=yaw)
        bbox = BoundingBox(center=[float(cen[0]), float(cen[1]), float(cen[2])],
                           size=size_wlh, orientation=orientation)
        setattr(bbox, 'rot', rot)
        return bbox

    # ---------------------
    # 兼容接口（占位）
    # ---------------------
    def num_frames(self):
        pass

    def num_tracklets(self):
        return len(self.annotations)

    def num_tracklet_frames(self, tracklet_id):
        pass

    def get_frame(self, tracklet_id, frame_id):
        pass

    def get_frame_bbox(self, tracklet_id, frame_id):
        if self.tracklets:
            frame = self.search_annotations[tracklet_id]['frames'][frame_id]
        else:
            frame_anno = self.annotations[tracklet_id][frame_id]
            frame = self._build_frame(frame_anno, bbox_only=True)
        return frame['pcd_bbox']


# ------------------------------------------------------------
# 可视化组合函数（保持原逻辑）
# ------------------------------------------------------------
def visualize_frames_and_boxes(template_img, template_img_bbox, search_img, search_img_bbox,
                               save_path=None, show_confidence=True, pred_bbox=None,
                               batch_id=None, scene_id=None, frame_id=None, confidence=None):
    """
    可视化模板帧与搜索帧以及对应 bbox。
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    matplotlib.use('Agg')

    def convert_bbox_format(bbox):
        if isinstance(bbox, dict):
            return [bbox.get('x_min', 0), bbox.get('y_min', 0),
                    bbox.get('x_max', 100), bbox.get('y_max', 100)]
        elif isinstance(bbox, (list, tuple)):
            return list(bbox)[:4]
        elif isinstance(bbox, torch.Tensor):
            return bbox.cpu().numpy().tolist()[:4]
        elif isinstance(bbox, np.ndarray):
            return bbox.tolist()[:4]
        else:
            return [0, 0, 100, 100]

    def convert_image_format(img):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return img.cpu().numpy()
        elif isinstance(img, Image.Image):
            return np.array(img)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise ValueError(f"Unsupported image format: {type(img)}")

    template_img_np = convert_image_format(template_img)
    search_img_np = convert_image_format(search_img)
    template_bbox_list = convert_bbox_format(template_img_bbox)
    search_bbox_list = convert_bbox_format(search_img_bbox)

    if template_img_np.max() <= 1.0:
        template_img_np = (template_img_np * 255).astype(np.uint8)
    if search_img_np.max() <= 1.0:
        search_img_np = (search_img_np * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(template_img_np)
    axes[0].set_title('Template Frame', fontsize=14, fontweight='bold', color='blue')
    axes[0].axis('off')

    if template_bbox_list and len(template_bbox_list) == 4:
        x_min, y_min, x_max, y_max = template_bbox_list
        if x_max > x_min and y_max > y_min:
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=3, edgecolor='blue', facecolor='none',
                label='Template GT'
            )
            axes[0].add_patch(rect)
            axes[0].text(x_min, y_min - 10, f'({x_min:.1f}, {y_min:.1f})',
                         color='blue', fontsize=10, weight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[1].imshow(search_img_np)
    axes[1].set_title('Search Frame', fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')

    if search_bbox_list and len(search_bbox_list) == 4:
        x_min, y_min, x_max, y_max = search_bbox_list
        if x_max > x_min and y_max > y_min:
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=3, edgecolor='green', facecolor='none',
                label='Search GT'
            )
            axes[1].add_patch(rect)

    if pred_bbox is not None:
        pred_bbox_list = convert_bbox_format(pred_bbox)
        if len(pred_bbox_list) == 4:
            x_min, y_min, x_max, y_max = pred_bbox_list
            if x_max > x_min and y_max > y_min:
                pred_color = 'orange' if confidence and confidence > 0.5 else 'red'
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=3, edgecolor=pred_color, facecolor='none',
                    linestyle='--', label=f'Prediction (conf: {confidence:.3f})' if confidence else 'Prediction'
                )
                axes[1].add_patch(rect)

    if show_confidence and confidence is not None:
        conf_color = 'green' if confidence > 0.9 else 'orange' if confidence > 0.5 else 'red'
        axes[1].text(0.02, 0.98, f'Confidence: {confidence:.3f}',
                     transform=axes[1].transAxes, fontsize=12, weight='bold',
                     color=conf_color, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    info_text = []
    if batch_id is not None:
        info_text.append(f'Batch: {batch_id}')
    if scene_id is not None:
        info_text.append(f'Scene: {scene_id}')
    if frame_id is not None:
        info_text.append(f'Frame: {frame_id}')
    if info_text:
        fig.suptitle(' | '.join(info_text), fontsize=12, y=0.02)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)

    if save_path is None:
        if batch_id is not None and scene_id is not None:
            vis_dir = f'vis/batch{batch_id}/Seq_{scene_id:06d}'
            os.makedirs(vis_dir, exist_ok=True)
            if frame_id is not None:
                save_path = f'{vis_dir}/{frame_id:05d}_comparison.jpg'
            else:
                save_path = f'{vis_dir}/comparison.jpg'
        else:
            os.makedirs('vis', exist_ok=True)
            save_path = 'vis/frames_comparison.jpg'

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Frame visualization saved to: {save_path}")
    plt.close()
    return save_path

