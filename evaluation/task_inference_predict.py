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
NMS_IOU = 0.65

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

    def run(self, model, config, device=None):
        # 如果没有传 device，就自动选择
        if device is None:
            device = select_device_auto()

        # 针对 4090 做额外优化（如果自动检测到）
        apply_4090_tuning_if_needed(device)

        # anno needed:['batch_id','scene_id','frame_id','contour']
        # search_tracklet_anno = self.annots['search']
        # template_frame_anno = self.annots['template']
        # anno needed:['batch_id','scene_id','frame_id','contour']
        search_tracklet_anno = [{'batch_id' : self.batch_id, 
                                 'scene': self.scene_id, 
                                 'frame': idx, 
                                 'contour': self.annots[0]['response_track'][idx - self.video_start_sec] if idx - self.video_start_sec < len(self.annots[0]['response_track']) else None} 
                                 for idx in range(self.video_start_sec, self.video_end_sec)] # 因为加上了template frame， 所以要-1
        template_frame_anno = {'batch_id' : self.batch_id,
                                'scene': self.scene_id,
                                'frame': self.annots[0]['query_frame'],
                                'contour': self.annots[0]['visual_crop']}
        # build frames
        search_frames = [self._build_search_frame(search_tracklet_anno[i]) for i in range(len(search_tracklet_anno))]
        template_frame = self._build_template_frame(template_frame_anno)
        # 将模板帧插入搜索帧序列结尾
        search_frames.append(template_frame)
        data = self._generate_item(template_frame, search_frames)

        query_set = self.annots[0]["metadata"]["query_set"]
        annot_key = f"{self.annotation_uid}_{query_set}"
        save_path = os.path.join(self.config.output_dir,self.config.dataset.name,"infer_outputs/like_ego4d", f'{annot_key}.pt')

        ret_bboxes, ret_scores = inference_video(config, model, data, save_path, device)

        # ensure output dir exists
        out_dir = osp.dirname(save_path)
        os.makedirs(out_dir, exist_ok=True)

        save_dict = {'ret_bboxes': ret_bboxes,
                     'ret_scores': ret_scores}
        torch.save(save_dict, save_path)

    def _build_search_frame(self, frame_anno: Dict[str, Any], bbox_only: bool = False) -> Dict[str, Any]:
        scene_id = frame_anno['scene']
        frame_id = frame_anno['frame']
        batch_id = int(frame_anno['batch_id'])
        frame_info = frame_anno['contour']

        pcd_folder = self._get_pcd_folder_path(batch_id, scene_id, 'o_s')
        pcd_file_dir = osp.join(
            self.data_dir, f"batch{batch_id}", 'img', 'o_s', 'Seq_%06d' % scene_id, f'{pcd_folder}/%05d.pcd' % frame_id)

        pcd_o3d = o3d.io.read_point_cloud(pcd_file_dir)
        pts = np.asarray(pcd_o3d.points, dtype=np.float32)
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
                rgb_bbox, intrinsic, extrinsic = self.prepare_vis_lists(rgb, camera_file_dir, frame_info)
                pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
                lable = torch.tensor([1], dtype=torch.int64)

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
        if not os.path.exists(pcd_file_dir):
            # 自动选用文件下的frame_id最小的pcd文件
            base_path = osp.join(
                self.data_dir, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, f'{pcd_folder}')
            pcd_files = [f for f in os.listdir(base_path) if f.endswith('.pcd')]
            if pcd_files:
                frame_id = min([int(f.split('.')[0]) for f in pcd_files])
            else:
                frame_id = 0
        pcd_file_dir = osp.join(
            self.data_dir, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, f'{pcd_folder}/%05d.pcd' % frame_id)
        pcd_o3d = o3d.io.read_point_cloud(pcd_file_dir)
        pts = np.asarray(pcd_o3d.points, dtype=np.float32)
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
            rgb_bbox, intrinsic, extrinsic = self.prepare_vis_lists(rgb, camera_file_dir, frame_info)
            pcd_bbox = self._bbox_from_contour(frame_anno.get('contour', None))
            # if rgb_bbox['x_min'] == rgb_bbox['x_max'] or rgb_bbox['y_min'] == rgb_bbox['y_max']:
            #     visualize_projection(rgb, rgb_bbox)

        return {'pcd': pcd, 'img': rgb, 'pcd_bbox': pcd_bbox, 'img_bbox': rgb_bbox, 'anno': frame_anno, 'intrinsic': intrinsic, 'extrinsic': extrinsic}
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
    def _find_matching_json_file(self, pattern_path: str) -> Optional[str]:
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
    def prepare_vis_lists(self, img: Image.Image, calib_file: str, frame_info: Dict[str, Any], point_size: float = 0.5) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
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
        calib_infos = self.parse_calib_file(calib_file)
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
    def parse_calib_file(self, calib_path: str) -> Dict[str, Dict[str, np.ndarray]]:
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

    def _bbox_from_contour(self, frame_info: Optional[Dict[str, Any]]):
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

        # vis(data)

        return data
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
        if self.config.dataset.query_square:
            bbox = dataset_utils.create_square_bbox(bbox.tolist(), h0, w0)

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

    def _process_clip(self, clip: torch.Tensor, clip_img_bbox: torch.Tensor, clip_with_bbox: torch.Tensor):
        """
        对抽取的视频帧与其 bbox 做 pad/resize，并尝试抽取 query_img。
        """
        target_size = self.config.dataset.clip_size_fine

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

    def process_pcd_bbox(self, pcd_bbox) -> torch.Tensor:
        """
        将 BoundingBox 转为 [cx, cy, cz, l, w, h, roll, pitch, yaw]。
        """
        center = pcd_bbox.center
        wlh = pcd_bbox.wlh
        rot = pcd_bbox.rot
        # 设置最小尺寸阈值，避免过小
        min_size = 0.01
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
        pcd_crop = crop_pcd_axis_aligned(
                pcd, bbox
            )
        pcd_rs, _ = resample_pcd(pcd, self.config.dataset.frame_npts, return_idx=True, is_training=True)
        pts3 = pcd_rs.points[:3, :].T  # (N, 3)
        return torch.as_tensor(pts3, dtype=torch.float32)

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

    def change_xyxy_to_yxyx(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        将 bbox 从 [x1,y1,x2,y2] 转为 [y1,x1,y2,x2]。
        """
        if len(bbox.shape) > 1:
            return bbox[:, [1, 0, 3, 2]]
        else:
            return torch.tensor([bbox[1], bbox[0], bbox[3], bbox[2]])

    def _get_query_(self, comp_template: Dict[str, Any]):
        """
        从模板帧生成 query_img 和其归一化 bbox。
        额外返回模板帧原始尺寸 (orig_h, orig_w)，用于相机投影映射。
        """
        target_size = self.config.dataset.clip_size_fine
        query_img = comp_template['img']                 # PIL
        bbox_dict = comp_template['img_bbox']
        bbox = torch.tensor([
            bbox_dict.get('x_min', 0.0),
            bbox_dict.get('y_min', 0.0),
            bbox_dict.get('x_max', 0.0),
            bbox_dict.get('y_max', 0.0),
        ], dtype=torch.float32)

        w0, h0 = query_img.size  # PIL: (W,H) 原始尺寸
        if self.config.dataset.query_square:
            bbox = dataset_utils.create_square_bbox(bbox.tolist(), h0, w0)

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


def inference_video(config, model, data, save_path, device):
    '''
    Perform VQ2D inference with improved device/data handling for modern NVIDIA GPUs (e.g. 4090).
    '''
    # get query
    # 注意：不要在循环里原地修改 data 中的 tensor，先 copy
    query_img = data['query'].clone() if isinstance(data['query'], torch.Tensor) else data['query']
    query_frame_bbox = data['query_frame_bbox'].clone() if isinstance(data['query_frame_bbox'], torch.Tensor) else data['query_frame_bbox']
    query_pcd = data['query_frame_pcd'].clone() if isinstance(data['query_frame_pcd'], torch.Tensor) else data['query_frame_pcd']
    query_frame_pcd_bbox = data['query_frame_pcd_bbox'].clone() if isinstance(data['query_frame_pcd_bbox'], torch.Tensor) else data['query_frame_pcd_bbox']

    clips_img = data['clip']    # list or tensor [T,3,H,W]
    clips_pcd = data['clip_pcd']  # [T,N,3]
    clip_num_frames = config.dataset.clip_num_frames    # 30
    batch_size = config.train.batch_size
    batch_num_frames = clip_num_frames * batch_size
    T = len(data['clip'])
    inference_time = (T - 1) // batch_num_frames
    if (T - 1) % batch_num_frames != 0:
        inference_time += 1

    # decide whether to use fp16 autocast (优先使用 config，如果没有则根据 device 判断)
    use_fp16 = False
    if hasattr(config, 'hardware') and getattr(config.hardware, 'force_fp16_inference', None) is not None:
        use_fp16 = bool(config.hardware.force_fp16_inference)
    else:
        use_fp16 = (device.type == 'cuda')  # 默认在 GPU 上启用 autocast(FP16)

    # 是否启用 channels_last (对 CNN 推理有利)
    use_channels_last = False
    if hasattr(config, 'hardware') and getattr(config.hardware, 'use_channels_last', None) is not None:
        use_channels_last = bool(config.hardware.use_channels_last)
    else:
        use_channels_last = (device.type == 'cuda')

    ret_bboxes, ret_scores = [], []
    for i in range(inference_time):
        idx_start = min(i * batch_num_frames, T-1)
        idx_end = min((i+1) * batch_num_frames, T-1)
        num_frames = idx_end - idx_start
        if num_frames < batch_num_frames:
            num_frames += 1
        batch_size_inference = num_frames // clip_num_frames
        if num_frames % clip_num_frames != 0:
            batch_size_inference += 1
        assert batch_size_inference <= batch_size

        inference_num_frames = batch_size_inference * clip_num_frames
        frame_idx = list(range(idx_start, idx_end))
        if len(frame_idx) < inference_num_frames:
            num_pad = inference_num_frames - len(frame_idx)
            frame_idx.extend([idx_end] * num_pad)   # pad with last index
            num_frames = len(frame_idx)

        # get current clips: load_clip 应返回 tensors
        clips_origin, clips_i, clips_p = load_clip(config, clips_img, clips_pcd, frame_idx)    # clips_i: [N,3,H,W]
        clips_origin = clips_origin[:num_frames]

        # rearrange到 [b,t,c,h,w]
        clips_i = clips_i[:num_frames]
        clips_i = rearrange(clips_i, '(b t) c h w -> b t c h w', b=batch_size_inference, t=clip_num_frames)

        # 处理输入类型与 device
        # 将图像转换为 channels_last if beneficial
        # if use_channels_last and isinstance(clips_i, torch.Tensor):
        #     if clips_i.ndim == 4:
        #         clips_i = clips_i.contiguous(memory_format=torch.channels_last)
        #     elif clips_i.ndim == 5:
        #         # 对 [b, t, c, h, w]，我们可以暂时展平再改回来
        #         b, t, c, h, w = clips_i.shape
        #         clips_i = clips_i.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        #         clips_i = clips_i.reshape(b, t, c, h, w)
        # to device / float (不在这里强制转为 fp16，交给 autocast 控制)
        clips_i = clips_i.to(device=device, non_blocking=True)
        query_img_to_device = query_img.to(device=device, non_blocking=True)

        # clone raw for visualization/debug
        clips_raw = clips_i.clone()
        query_raw = query_img_to_device.clone()

        # process inputs (你的 process_inputs 应该接受 device 上的张量)
        # 为了兼容 autocast，我们在 with autocast block 中调用 model forward
        try:
            clips_i_proc, query_i_proc = process_inputs(clips_i, query_img_to_device)
        except Exception as e:
            # 如果 process_inputs 期望 CPU 张量或其他格式，这里回退到原始方式（以便调试）
            raise RuntimeError(f"process_inputs failed: {e}")

        # ensure dims for pcd/bbox
        if isinstance(query_pcd, torch.Tensor) and query_pcd.dim() == 2:
            query_pcd = query_pcd.unsqueeze(0)
        if isinstance(query_frame_bbox, torch.Tensor) and query_frame_bbox.dim() == 1:
            query_frame_bbox = query_frame_bbox.unsqueeze(0)
        if isinstance(query_frame_pcd_bbox, torch.Tensor) and query_frame_pcd_bbox.dim() == 1:
            query_frame_pcd_bbox = query_frame_pcd_bbox.unsqueeze(0)

        # move pcd/bbox to device
        clips_p = clips_p.to(device=device, non_blocking=True).float()
        query_pcd = query_pcd.to(device=device, non_blocking=True).float()
        query_frame_bbox = query_frame_bbox.to(device=device, non_blocking=True).float()
        query_frame_pcd_bbox = query_frame_pcd_bbox.to(device=device, non_blocking=True).float()

        # inference with autocast (mixed precision) if available/desired
        with torch.no_grad():
            # note: 指定 dtype 到 autocast 可以帮助确保使用 FP16 on supported GPUs
            if device.type == 'cuda' and use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32

            with autocast(dtype=dtype):
                preds = model(clips_i_proc, query_i_proc, query_frame_bbox, clips_p, query_pcd, query_frame_pcd_bbox,
                              fix_backbone=getattr(config.model, 'fix_backbone', False))

        preds_top = get_top_predictions(config, preds, num_frames)
        ret_bboxes.append(preds_top['bbox'])
        ret_scores.append(preds_top['prob'])

        if getattr(config, 'debug', False):
            try:
                vis_utils.vis_pred_clip_inference(clips=clips_origin,
                                                 queries=query_raw,
                                                 pred=preds_top,
                                                 save_path=save_path,
                                                 iter_num=i)
            except Exception:
                # 可视化失败不应中断推理
                pass

    # concat results
    assert len(ret_bboxes) > 0
    ret_bboxes = torch.cat(ret_bboxes, dim=0)
    ret_scores = torch.cat(ret_scores, dim=0)
    # 截断到原始长度
    ret_bboxes = ret_bboxes[:T-1]
    ret_scores = ret_scores[:T-1]
    return ret_bboxes, ret_scores


def get_top_predictions(config, preds, num_frames):
    """
    与原来逻辑一致，但确保返回 cpu 张量且长度截断到 num_frames
    """
    resize_res = config.dataset.clip_size_coarse

    pred_center = preds['center']
    pred_size = preds['size']
    pred_rot = preds['rot']
    pred_bbox = preds['bbox']
    pred_prob = preds['prob']

    if len(pred_prob.shape) == 3:
        b,t,N = pred_prob.shape
        pred_prob = rearrange(pred_prob, 'b t N -> (b t) N')
        pred_size = rearrange(pred_size, 'b t N c -> (b t) N c')
        pred_center = rearrange(pred_center, 'b t N c -> (b t) N c')
        pred_rot = rearrange(pred_rot, 'b t N c -> (b t) N c')
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c')
        pred_prob_all = pred_prob.clone()

        pred_prob, top_idx = torch.max(pred_prob, dim=-1)  # [b*t]
        # gather top predictions
        top_idx_expand = top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pred_bbox.shape[-1])
        pred_bbox = torch.gather(pred_bbox, dim=1, index=top_idx_expand).squeeze(1)
        # pred_size / center / rot if needed (与原逻辑类似)
        # 但要确保 squeeze 后维度正确
    else:
        b,t = pred_prob.shape
        pred_prob = rearrange(pred_prob, 'b t -> (b t)')
        pred_size = rearrange(pred_size, 'b t c -> (b t) c')
        pred_center = rearrange(pred_center, 'b t c -> (b t) c')
        pred_rot = rearrange(pred_rot, 'b t c -> (b t) c')
        pred_bbox = rearrange(pred_bbox, 'b t c -> (b t) c')

    pred_prob_raw = pred_prob.clone().detach().cpu()
    pred_bbox_raw = pred_bbox.clone().detach().cpu()
    pred_prob = pred_prob[:num_frames]
    pred_bbox = pred_bbox[:num_frames]

    preds_out = {
        'bbox_raw': pred_bbox_raw,
        'prob_raw': pred_prob_raw,
        'bbox': pred_bbox.detach().cpu(),
        'prob': pred_prob.detach().cpu(),
    }
    return preds_out