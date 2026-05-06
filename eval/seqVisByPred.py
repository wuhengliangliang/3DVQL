# -*- coding: utf-8 -*-
# @Time    : 2024/9/30
# @Author  : Yifan JIAO
# @Project : SOT_Dataset
# @File    : seqVis.py

"""
    将目标序列的标注结果(txt)可视化，导出为图片，并储存成视频
"""

import json
import cv2
import math
import os
import glob
import open3d as o3d
import numpy as np
import matplotlib
# 使用无界面后端，确保SSH环境可渲染保存
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 触发3D投影注册
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from transforms3d.euler import euler2mat

from utils.mathConvert import (compute_3d_box_vertices, project_3d_to_2d,
                         compute_rotation_matrix_from_directions)
from .utils.vis import parse_label_file_txt, parse_label_file_from_prediction

order = 'rxyz'  # euler2angle rotation order
z_p = 0.005
# 测试数据根目录：自动遍历 batch*/img/o_s/Seq_*
TEST_ROOT_DIR = "/data_0/pl/VQL_Data/VQL_Data_test"
save_output_dir = "output/vis_results"  # SSH环境下保存图片的目录
# 在SSH/无显示环境下强制使用离屏渲染（不创建Open3D窗口）
HEADLESS_MODE = True
# 输出布局：overlay(右上角叠加) 或 side_by_side(左右排列，避免重叠)
IMAGE_LAYOUT = 'side_by_side'

# 颜色方案：多方法 + GT(红色)
METHOD_COLORS_RGB = [
    [0, 1, 0],      # 绿色
    [0, 0, 1],      # 蓝色
    [1, 0.65, 0],   # 橙色
    [1, 0, 1],      # 品红
    [0, 1, 1],      # 青色
    [1, 1, 1],      # 白色
    [1, 0, 0]       # 红色 (GT)
]


class PointCloudSequenceViewer:
    def __init__(self, seq_id_, pcd_files, img_files, gt_info_, line_set_list_, save_dir):
        self.seq_id = seq_id_
        self.pcd_files = pcd_files
        self.img_files = img_files
        self.frame_infos = gt_info_
        self.line_set_list = line_set_list_
        self.save_dir = save_dir
        
        # 创建保存目录
        import os
        os.makedirs(self.save_dir, exist_ok=True)

        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(visible=False, width=1400, height=1080)

        # 加载第一个点云和图像
        self.pcd = self.pcd_files[self.index]
        self.img = self.img_files[self.index]
        self.vis.add_geometry(self.pcd)
        
        # 添加所有方法的边界框
        if self.frame_infos[self.index]['occlusion'] == 0:
            for i in range(len(self.line_set_list[self.index])):
                self.vis.add_geometry(self.line_set_list[self.index][i])
        self.vis.add_geometry(self.img)

        # 设置视图控制（SSH环境下可能无效，但不会报错）
        set_control(
            self.vis, self.frame_infos[self.index]['center3D'], self.frame_infos[0]['size3D'], point_size=1.5)
        
        # SSH环境下需要手动更新一次渲染
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
        except:
            pass

        # 绑定按键
        self.vis.register_key_callback(ord(","), self.previous_pcd)
        self.vis.register_key_callback(ord("."), self.next_pcd)
        self.vis.register_key_callback(ord("S"), self.save_current_frame)
        self.vis.register_key_callback(ord("A"), self.save_all_frames)

    def previous_pcd(self, vis):
        if self.index > 0:
            self.index -= 1
            self.update_pcd()
        return False

    def next_pcd(self, vis):
        if self.index < len(self.pcd_files) - 1:
            self.index += 1
            self.update_pcd()
        return False

    def save_current_frame(self, vis):
        """保存当前帧"""
        import os
        save_path = os.path.join(self.save_dir, f"frame_{self.index:05d}.png")
        self.vis.capture_screen_image(save_path, do_render=True)
        print(f"Saved frame {self.index} to {save_path}")
        return False

    def save_all_frames(self, vis):
        """保存所有帧"""
        import os
        print(f"Saving all {len(self.pcd_files)} frames...")
        for i in tqdm(range(len(self.pcd_files))):
            self.index = i
            self.update_pcd()
            save_path = os.path.join(self.save_dir, f"frame_{i:05d}.png")
            self.vis.capture_screen_image(save_path, do_render=True)
        print(f"All frames saved to {self.save_dir}")
        return False

    def update_pcd(self):
        self.vis.clear_geometries()
        self.pcd = self.pcd_files[self.index]
        self.img = self.img_files[self.index]

        self.vis.add_geometry(self.pcd)
        if self.frame_infos[self.index]['occlusion'] == 0:
            for i in range(len(self.line_set_list[self.index])):
                self.vis.add_geometry(self.line_set_list[self.index][i])
        self.vis.add_geometry(self.img)

        set_control(
            self.vis, self.frame_infos[self.index]['center3D'], self.frame_infos[0]['size3D'], point_size=1.5)
        
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
        except:
            pass

    def run(self):
        """SSH环境下直接保存所有帧"""
        print(f"Running in SSH mode - saving all frames to {self.save_dir}")
        self.save_all_frames(self.vis)
        self.vis.destroy_window()
        
    def run_interactive(self):
        """交互模式（仅在有GUI环境时使用）"""
        self.vis.run()
        self.vis.destroy_window()


def parse_label_file(filepath):
    """
    Parse the JSON file to extract required information from each frame.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A list of dictionaries, each containing information for one frame.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames_info = []

    for frame_ in data:
        for obj in frame_.get('objects', []):
            if obj.get('type') == '3D_BOX':
                class_name = obj.get('className')
                class_values = obj.get('classValues', [])
                occlusion_value = None
                out_of_view_value = None
                for value in class_values:
                    if value.get('alias') == 'occlusion':
                        occlusion_value = value.get('value')
                    elif value.get('alias') == 'out-of-view':
                        out_of_view_value = value.get('value')

                contour = obj.get('contour', {})
                point_n = contour.get('pointN')

                size3d_dict = contour.get('size3D', {})
                size3d = [
                    size3d_dict.get('x', 0),
                    size3d_dict.get('y', 0),
                    size3d_dict.get('z', 0)
                ]

                center3d_dict = contour.get('center3D', {})
                center3d = [
                    center3d_dict.get('x', 0),
                    center3d_dict.get('y', 0),
                    center3d_dict.get('z', 0)
                ]

                rotation3d_dict = contour.get('rotation3D', {})
                rotation3d = [
                    rotation3d_dict.get('x', 0),
                    rotation3d_dict.get('y', 0),
                    rotation3d_dict.get('z', 0)
                ]

                frame_info = {
                    'ClassName': class_name,
                    'occlusion': occlusion_value,
                    'out-of-view': out_of_view_value,
                    'pointN': point_n,
                    'size3D': size3d,
                    'center3D': center3d,
                    'rotation3D': rotation3d
                }

                frames_info.append(frame_info)

    return frames_info


def parse_calib_file(calib_path):
    """
        Parse camera parameters from a JSON file.

        Args:
            calib_path (str): Path to the JSON file containing camera parameters.

        Returns:
            dict: Dictionary containing camera parameters for each camera.
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

        cameras[f'camera_{idx}'] = {
            'intrinsic': intrinsic,
            'extrinsic': extrinsic
        }

    return cameras


def create_arrow(origin, rot_matrix, frame_info, color=(1, 0, 0)):
    """创建一个指向给定方向的箭头"""
    length = frame_info["size3D"][0] / 2.0
    radius = min(frame_info["size3D"][0], frame_info["size3D"][1]) / 10.0
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=1.5 * radius,
        cone_height=4.0 * radius,
        cylinder_radius=radius,
        cylinder_height=length
    )
    mesh_arrow.paint_uniform_color(color)

    # 计算从默认方向z到目标方向x的旋转矩阵
    default_direction = np.array([0, 0, 1])  # z方向
    target_direction = np.array([1, 0, 0])  # x方向
    r_default_to_target = compute_rotation_matrix_from_directions(default_direction, target_direction)
    mesh_arrow.rotate(rot_matrix @ r_default_to_target, center=(0, 0, 0))
    mesh_arrow.translate(origin)
    return mesh_arrow


def create_lineset_from_box(vertices, color):
    """
    Create a LineSet for the 3D bounding box from its vertices.

    Args:
        vertices: A numpy array of shape (8, 3) representing the 8 vertices of the 3D bounding box.

    Returns:
        A LineSet object representing the 3D bounding box.
    """
    # lines = [
    #     [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
    #     [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
    #     [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges
    # ]
    lines = [
        [0, 1], [1, 5], [5, 4], [4, 7],  # Top square
        [3, 2], [2, 6], [6, 7], [7, 3],  # Bottom square
        [0, 3], [1, 2], [5, 6], [4, 7]  # Vertical edges
    ]

    colors = [color for _ in range(len(lines))]  # Red color for all edges

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


# =====================
# Headless 渲染辅助函数
# =====================
def _build_frame_vertices_and_overlay(seq_infos, i, intrinsic, extrinsic):
    """
    计算当前帧所有方法的3D框顶点，并生成右上角的2D叠加图像。
    返回:
      - vertices_list: List[np.ndarray(8,3)]
      - overlay_img: np.ndarray(H,W,3) RGB
    """
    # 读取并标注2D图像
    overlay_img = None
    vertices_list = []

    # 2D底图：从最后一个seq_infos[-1](GT)的遮挡信息判断是否绘制
    # 注意：调用方先读取原始img并传入，这里仅绘制3D框投影
    # 但为了复用现有流程，这里由调用方提供img更灵活。当前函数只构造顶点和返回标志。
    return vertices_list, overlay_img


def make_overlay_image(img, seq_id, frame_index, seq_infos, intrinsic, extrinsic):
    """
    基于输入图像，在上面绘制每个方法的3D框投影，返回叠加后的图像(RGB)。
    """
    img_draw = add_text_to_image(img, text=f"Seq {str(seq_id).zfill(5)} Frame {frame_index+1}", pos=(50, 60))

    # 统一遮挡判断：以GT为准（seq_infos最后一个为GT）
    try:
        occluded = seq_infos[-1][frame_index].get('occlusion', 0) != 0
    except Exception:
        occluded = False

    if occluded:
        # 遮挡：不画任何2D框，仅加标注
        img_draw = add_text_to_image(img_draw, "OCCLUSION", pos=(50, 120), font_color=(255, 255, 0))
        return img_draw

    # 非遮挡：绘制各方法2D投影
    for j, pred in enumerate(seq_infos):
        frame_pred = pred[frame_index]
        # 计算3D边界框顶点（支持欧拉角或旋转矩阵）
        if len(frame_pred['rotation3D']) == 9:
            rot_matrix = np.array(frame_pred['rotation3D']).reshape(3, 3)
            pred_box_vertices = compute_3d_box_vertices(
                frame_pred['center3D'], frame_pred['size3D'], rot_matrix)
        else:
            pred_box_vertices = compute_3d_box_vertices(
                frame_pred['center3D'], frame_pred['size3D'], frame_pred['rotation3D'])

        pred_box_vertices_2d = project_3d_to_2d(pred_box_vertices, intrinsic, extrinsic)
        color_bgr = tuple([int(c * 255) for c in METHOD_COLORS_RGB[j][::-1]])
        img_draw = draw_3d_bbox_projection(img_draw, pred_box_vertices_2d, color=color_bgr, thickness=2)
    return img_draw


def compute_all_vertices_for_frame(seq_infos, frame_index):
    """
    仅计算3D框顶点列表（用于3D可视化绘制线框）。
    返回 List[np.ndarray(8,3)]
    """
    # 若GT标注为遮挡，则不返回任何3D框顶点（与交互式可视化逻辑保持一致）
    try:
        if seq_infos[-1][frame_index]['occlusion'] != 0:
            return []
    except Exception:
        # 容错：若字段缺失或索引异常，则默认不遮挡
        pass

    vertices_list = []
    for j, pred in enumerate(seq_infos):
        frame_pred = pred[frame_index]
        if len(frame_pred['rotation3D']) == 9:
            rot_matrix = np.array(frame_pred['rotation3D']).reshape(3, 3)
            pred_box_vertices = compute_3d_box_vertices(
                frame_pred['center3D'], frame_pred['size3D'], rot_matrix)
        else:
            pred_box_vertices = compute_3d_box_vertices(
                frame_pred['center3D'], frame_pred['size3D'], frame_pred['rotation3D'])
        vertices_list.append(pred_box_vertices)
    return vertices_list


def render_3d_matplotlib(points, vertices_list, center, object_size,
                         fig_w=1400, fig_h=1080, dpi=100):
    """
    使用Matplotlib在SSH下绘制点云 + 3D线框，并返回RGB图像(np.ndarray HxWx3)。
    将2D叠加由外部进行合成。
    """
    # 创建图形
    fig = plt.figure(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云（灰色）
    if points is not None and len(points) > 0:
        pts = points
        # 控制点数以避免过慢
        if len(pts) > 80000:
            idx = np.random.choice(len(pts), 80000, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='0.5', s=0.5, alpha=0.8, depthshade=False)

    # 绘制边界框
    lines = [
        [0, 1], [1, 5], [5, 4], [4, 7],
        [3, 2], [2, 6], [6, 7], [7, 3],
        [0, 3], [1, 2], [5, 6], [4, 7]
    ]
    for j, vertices in enumerate(vertices_list):
        col = METHOD_COLORS_RGB[j]
        for a, b in lines:
            x = [vertices[a, 0], vertices[b, 0]]
            y = [vertices[a, 1], vertices[b, 1]]
            z = [vertices[a, 2], vertices[b, 2]]
            ax.plot(x, y, z, color=col, linewidth=1.5)

    # 视角与范围（模拟 set_control）
    z = (object_size[0] * object_size[1] * object_size[2]) ** (1.0 / 3.0)
    z *= z_p

    # 设置轴范围围绕目标中心
    cx, cy, cz = center
    # 简单按目标尺寸设定可视范围
    span = max(object_size) * 4.0
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - span, cz + span)

    # 依据前向向量[-1,0,0.5]估算视角
    # elev ~ arcsin(z), azim ~ atan2(y, x)
    elev = np.degrees(np.arcsin(0.5 / np.sqrt(1.0 + 0.5 * 0.5)))  # 约26.565°
    azim = 180.0  # 朝向 -X
    ax.view_init(elev=elev, azim=azim)

    # 外观设置
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])

    fig.tight_layout(pad=0)
    # 渲染为图像
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return np.ascontiguousarray(img)


def paste_overlay(top_img, overlay_img, scale=0.5, margin_ratio=0.02):
    """将overlay_img按比例缩放(保持原始宽高比)并贴到top_img右上角。返回合成后的RGB图。"""
    if overlay_img is None:
        return top_img
    H, W, _ = top_img.shape
    max_w = max(1, int(W * scale))
    max_h = max(1, int(H * scale))

    oh, ow = overlay_img.shape[0], overlay_img.shape[1]
    # 计算等比例缩放系数，限制到最大框内
    s = min(max_w / float(ow), max_h / float(oh))
    new_w = max(1, int(round(ow * s)))
    new_h = max(1, int(round(oh * s)))

    # OpenCV resize 要求BGR，这里转换两次，保持RGB一致
    overlay_resized = cv2.resize(overlay_img[:, :, ::-1], (new_w, new_h))[:, :, ::-1]

    m = int(min(H, W) * margin_ratio)
    y0 = m
    x0 = W - new_w - m
    # 边界防护
    y0 = max(0, min(y0, H - new_h))
    x0 = max(0, min(x0, W - new_w))

    out = top_img.copy()
    out[y0:y0 + new_h, x0:x0 + new_w] = overlay_resized
    return out


def compose_side_by_side(left_img, right_img, right_scale=0.5, gap=10, bg_color=(0, 0, 0)):
    """
    将两张RGB图像左右排列，右图按 right_scale 相对左图宽度缩放，保持同高，避免重叠。
    Args:
        left_img: 左侧RGB图(如3D渲染)
        right_img: 右侧RGB图(如2D投影)
        right_scale: 右图相对左图宽度比例（默认0.5）
        gap: 左右图之间的像素间距
        bg_color: 背景色RGB
    Returns:
        合成后的RGB图
    """
    if right_img is None:
        return left_img
    H, W, _ = left_img.shape
    max_h = H
    max_w = max(1, int(W * right_scale))
    rh, rw = right_img.shape[0], right_img.shape[1]
    # 按比例缩放，限制在(max_w, max_h)内
    s = min(max_w / float(rw), max_h / float(rh))
    new_w = max(1, int(round(rw * s)))
    new_h = max(1, int(round(rh * s)))
    # resize 保持宽高比
    right_resized = cv2.resize(right_img[:, :, ::-1], (new_w, new_h))[:, :, ::-1]

    out_h = H  # 以左图高度为准，右图居中贴合
    out_w = W + gap + new_w
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[:, :, 0] = int(bg_color[0] * 255)
    out[:, :, 1] = int(bg_color[1] * 255)
    out[:, :, 2] = int(bg_color[2] * 255)

    out[:, :W] = left_img
    y0 = (H - new_h) // 2
    out[y0:y0 + new_h, W + gap: W + gap + new_w] = right_resized
    return out


def prepare_vis_lists(pcd_file, img_file, calib_file, seq_infos, i, point_size=0.5):
    """
    可视化点云和3D边界框，在SSH环境下只生成数据不显示
    """
    line_sets = []
    
    # Load the calibration data
    calib_infos = parse_calib_file(calib_file)
    intrinsic_camera_0, extrinsic_camera_0 = calib_infos['camera_0']['intrinsic'], calib_infos['camera_0']['extrinsic']
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # 设置点云颜色为浅灰色
    points = np.asarray(pcd.points)
    colors = np.ones((len(points), 3)) * 0.5  # 灰色点云
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Load the 2D image
    img = plt.imread(img_file)
    # 注意：此函数在 GUI 模式下使用，外部已有 seq_id 变量；批处理路径默认使用 headless，不再依赖此文本
    img = add_text_to_image(img, text=f"Frame {i+1}", pos=(50, 60))

    # 定义每个方法的颜色
    colors_bbox = [
        [0, 1, 0],      # 绿色
        [0, 0, 1],      # 蓝色
        [1, 0.65, 0],   # 橙色
        [1, 0, 1],      # 品红
        [0, 1, 1],      # 青色
        [1, 1, 1],      # 白色
        [1, 0, 0]       # 红色 (GT)
    ]
    
    # 统一遮挡判断（以GT为准）
    try:
        occluded = seq_infos[-1][i].get('occlusion', 0) != 0
    except Exception:
        occluded = False

    # 处理每个方法的预测结果
    for j, pred in enumerate(seq_infos):
        frame_pred = pred[i]
        # 计算3D边界框的8个顶点
        if len(frame_pred['rotation3D']) == 9:
            rot_matrix = np.array(frame_pred['rotation3D']).reshape(3, 3)
            pred_box_vertices = compute_3d_box_vertices(
                frame_pred['center3D'], frame_pred['size3D'], rot_matrix)
        else:
            pred_box_vertices = compute_3d_box_vertices(
                frame_pred['center3D'], frame_pred['size3D'], frame_pred['rotation3D'])

        # 2D绘制：仅在非遮挡时画框
        if not occluded:
            pred_box_vertices_2d = project_3d_to_2d(pred_box_vertices, intrinsic_camera_0, extrinsic_camera_0)
            color_bgr = tuple([int(c * 255) for c in colors_bbox[j][::-1]])
            img = draw_3d_bbox_projection(img, pred_box_vertices_2d, color=color_bgr, thickness=2)

        # 创建3D线框（GUI路径中是否添加由上层控制）
        line_set = create_lineset_from_box(pred_box_vertices, colors_bbox[j])
        line_sets.append(line_set)

    # 遮挡时添加一次文本标记
    if occluded:
        img = add_text_to_image(img, "OCCLUSION", pos=(50, 120), font_color=(255, 255, 0))

    # 创建Matplotlib图形用于2D图像
    fig = Figure(figsize=(14, 10.8), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    fig.tight_layout(pad=0)

    # 调整子图位置到右上角
    ax_position = ax.get_position()
    ax.set_position([ax_position.x0 + 0.5, ax_position.y0 + 0.5, 
                     ax_position.width / 2, ax_position.height / 2])

    # 渲染Matplotlib图形为图像
    canvas.draw()
    img_array = np.array(canvas.renderer.buffer_rgba())

    # 转换为RGB格式
    img_array = img_array[:, :, :3]
    img_array = np.ascontiguousarray(img_array)

    # 转换为Open3D图像格式
    o3d_img = o3d.geometry.Image(img_array)

    return pcd, line_sets, o3d_img


def set_control(vis, center, object_size, point_size=0.5):
    z = (object_size[0] * object_size[1] * object_size[2]) ** (1 / 3)
    # z = math.exp(-z) * 0.25  # 单调递减，值域为(0, 1)，这样zoom后，大目标的场景更大，小目标的场景更小
    z *= z_p
    # print(z)

    # 在SSH环境下，get_render_option()可能返回None，需要检查
    try:
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = point_size
    except:
        pass  # SSH环境下可能无法设置渲染选项

    # Set the view control to adjust the camera
    try:
        view_ctl = vis.get_view_control()
        if view_ctl is not None:
            view_ctl.set_front([-1, 0, 0.5])  # Camera looking direction
            view_ctl.set_lookat(center)  # Center of the view
            view_ctl.set_up([0, 0, 1])  # Camera's up direction
            view_ctl.set_zoom(z)  # Adjust zoom as needed
    except:
        pass  # SSH环境下可能无法设置视图控制


def are_points_in_box(points, center, dimensions, rotation):
    """
    Determine if points are inside a 3D bounding box.

    Args:
        points (np.ndarray): An Nx3 array representing the coordinates of N points.
        center (array-like): A 3-element array representing the center of the box.
        dimensions (array-like): A 3-element array representing the dimensions (length, width, height) of the box.
        rotation (array-like): A 3-element array representing the rotation angles (yaw, pitch, roll) in radians.

    Returns:
        np.ndarray: A boolean array of length N, where True indicates the corresponding point is inside the box.
    """
    # Compute the rotation matrix
    # r = compute_rotation_matrix(rotation)
    r = euler2mat(rotation[0], rotation[1], rotation[2], order)

    # Translate points to the box's local coordinate system
    local_points = (points - center) @ r

    # Define the half-dimensions of the box
    half_dim = np.array(dimensions) / 2.0

    # Check if points are within the box's bounds
    inside_mask = np.all(np.abs(local_points) <= half_dim, axis=1)

    return inside_mask


def draw_3d_bbox_projection(image, bbox_2d_points, color=(0, 255, 0), thickness=2):
    """
    Draw the projection of a 3D bounding box on a 2D image.

    Args:
        image (numpy.ndarray): Input image.
        bbox_2d_points (np.ndarray): Nx2 array of 2D points representing the projection of 3D bounding box.
        color (tuple): Color of the bounding box lines in BGR format (default is green).
        thickness (int): Thickness of the bounding box lines (default is 2).

    Returns:
        numpy.ndarray: Image with the 3D bounding box projection drawn.
    """
    bbox_2d_points = np.array(bbox_2d_points)
    qs = bbox_2d_points.reshape(-1, 2).astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html

        # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制边界框的前四条边。
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制边界框的后四条边，与前四条边平行
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制连接前四条边和后四条边的边界框的边。
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return image


def add_text_to_image(image, text, pos=(10, 60), font_scale=2, font_color=(255, 0, 0), thickness=3):
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Add text to the image using OpenCV
    cv2.putText(image_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    # Convert back to RGB for matplotlib
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def create_video_from_frames(frame_dir, output_video_path, fps=10):
    """
    从保存的帧图像创建视频
    
    Args:
        frame_dir: 帧图像所在目录
        output_video_path: 输出视频路径
        fps: 视频帧率
    """
    import glob
    
    # 获取所有帧图像
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(frame_files)} frames...")
    for frame_file in tqdm(frame_files):
        frame = cv2.imread(frame_file)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    import os
    import re

    def _parse_seq_numeric_id(seq_name: str) -> int:
        try:
            m = re.search(r"(\d+)$", seq_name)
            return int(m.group(1)) if m else 0
        except Exception:
            return 0

    def list_test_sequences(root_dir):
        """扫描测试数据目录，返回 (batch_name, seq_name, seq_dir, calib_json) 列表。"""
        results = []
        if not os.path.isdir(root_dir):
            print(f"Error: TEST_ROOT_DIR not found: {root_dir}")
            return results
        for batch_name in sorted(os.listdir(root_dir)):
            batch_path = os.path.join(root_dir, batch_name)
            if not os.path.isdir(batch_path):
                continue
            img_root = os.path.join(batch_path, 'img', 'o_s')
            if not os.path.isdir(img_root):
                continue
            for seq_name in sorted(os.listdir(img_root)):
                seq_dir = os.path.join(img_root, seq_name)
                if not os.path.isdir(seq_dir) or not seq_name.startswith('Seq_'):
                    continue
                calib_json = os.path.join(seq_dir, f"{seq_name}.json")
                if not os.path.exists(calib_json):
                    print(f"Warning: calib not found, skip {batch_name}/{seq_name}")
                    continue
                results.append((batch_name, seq_name, seq_dir, calib_json))
        return results

    def process_sequence(batch_name, seq_name, seq_dir, calib_file_path):
        # 定义要比较的方法列表（可扩展）
        methods = [
            # 'MBPTrackModified9DoF2StageXYZ_mem2_jyfset',
            # 'mbptrack3d_jyfset',
            # 'm2_track_jyfset',
            # 'p2b_jyfset',
            # 'bat_jyfset',
            # 'm3sot_jyfset',
            # 'ptt_jyfset',
            # 'cxtrack3d_jyfset',
            # 'seqtrack3d_jyfset',
            'VQL7dof'
        ]

        seq_infos = []
        # 加载每个方法的预测结果
        for method in methods:
            txt_file_path = f"output/VQLOC/infer_outputs/{method}/{seq_name}.txt"
            if os.path.exists(txt_file_path):
                seq_frames_info = parse_label_file_from_prediction(txt_file_path)
                seq_infos.append(seq_frames_info)
                print(f"  Loaded predictions from {method}: {len(seq_frames_info)} frames")
            else:
                print(f"  Warning: {txt_file_path} not found, skipping {method}")

        # 加载Ground Truth（必须存在，否则跳过本序列）
        gt_file_path = f"output/VQLOC/infer_outputs/VQL7dof_gt/{seq_name}.txt"
        if os.path.exists(gt_file_path):
            gt_info = parse_label_file_txt(gt_file_path)
            seq_infos.append(gt_info)
            print(f"  Loaded GT: {len(gt_info)} frames")
        else:
            print(f"  Error: GT file {gt_file_path} not found, skip {batch_name}/{seq_name}")
            return False

        # 校验：至少应有1个方法 + GT
        if len(seq_infos) < 2:
            print(f"  Error: no prediction loaded for {seq_name}, skip")
            return False

        # 读取标定
        calib_infos = parse_calib_file(calib_file_path)
        K = calib_infos['camera_0']['intrinsic']
        E = calib_infos['camera_0']['extrinsic']

        # 帧数上限与对齐：取 min(200, GT长度, 各方法长度)
        lengths = [len(s) for s in seq_infos]
        num_frames = min(*lengths)
        if num_frames <= 0:
            print(f"  Warning: no frames to process for {seq_name}, skip")
            return False

        # 输出目录：按 batch/seq 划分，避免同名冲突
        save_dir = os.path.join(save_output_dir, batch_name, seq_name)
        if os.path.exists(save_dir):
            return True  # 已处理过，跳过
        os.makedirs(save_dir, exist_ok=True)

        # 数据子目录
        pcd_dir = os.path.join(seq_dir, 'lidar_point_cloud_0')
        img_dir = os.path.join(seq_dir, 'camera_image_0')
        if not os.path.isdir(pcd_dir) or not os.path.isdir(img_dir):
            print(f"  Error: missing pcd/img dir for {seq_name}, skip")
            return False

        sid = _parse_seq_numeric_id(seq_name)

        if HEADLESS_MODE:
            print(f"Headless: {batch_name}/{seq_name} -> {num_frames} frames ...")
            for i in tqdm(range(num_frames)):
                pcd_file_path = os.path.join(pcd_dir, f"{str(i + 1).zfill(5)}.pcd")
                img_file_path = os.path.join(img_dir, f"{str(i + 1).zfill(5)}.jpg")

                if not os.path.exists(pcd_file_path) or not os.path.exists(img_file_path):
                    print(f"  Warning: missing data for frame {i+1}, skipping.")
                    continue

                # 读取点云
                pcd = o3d.io.read_point_cloud(pcd_file_path)
                pts = np.asarray(pcd.points)

                # 计算所有方法的3D框
                vertices_list = compute_all_vertices_for_frame(seq_infos, i)

                # 生成2D叠加图
                raw_img = plt.imread(img_file_path)
                overlay_img = make_overlay_image(raw_img, sid, i, seq_infos, K, E)

                # 渲染3D（使用当前帧 GT 尺寸/中心）
                frame_img_3d = render_3d_matplotlib(pts, vertices_list, seq_infos[-1][i]['center3D'], seq_infos[-1][i]['size3D'])

                # 合成：根据布局选择叠加或左右排列
                if IMAGE_LAYOUT == 'side_by_side':
                    final_img = compose_side_by_side(frame_img_3d, overlay_img, right_scale=0.5, gap=20, bg_color=(0, 0, 0))
                else:
                    final_img = paste_overlay(frame_img_3d, overlay_img, scale=0.5, margin_ratio=0.02)

                save_path = os.path.join(save_dir, f"frame_{i:05d}.png")
                cv2.imwrite(save_path, final_img[:, :, ::-1])

            print(f"  Done: frames saved to {save_dir}")
            # 创建视频
            video_path = os.path.join(save_output_dir, batch_name, f"{seq_name}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            create_video_from_frames(save_dir, video_path, fps=10)
            print(f"  Video created: {video_path}")
        else:
            # GUI 模式（如需）
            pcd_list, line_set_list, o3d_img_list = [], [], []
            print(f"GUI: loading {num_frames} frames for {batch_name}/{seq_name} ...")
            for i in tqdm(range(num_frames)):
                pcd_file_path = os.path.join(pcd_dir, f"{str(i + 1).zfill(5)}.pcd")
                img_file_path = os.path.join(img_dir, f"{str(i + 1).zfill(5)}.jpg")
                if not os.path.exists(pcd_file_path) or not os.path.exists(img_file_path):
                    print(f"  Warning: missing data for frame {i+1}, skipping.")
                    continue
                cur_pcd, cur_line_set, cur_o3d_img = prepare_vis_lists(
                    pcd_file_path,
                    img_file_path,
                    calib_file_path,
                    seq_infos,
                    i,
                    point_size=1.5,
                )
                pcd_list.append(cur_pcd)
                line_set_list.append(cur_line_set)
                o3d_img_list.append(cur_o3d_img)

            print(f"  Loaded {len(pcd_list)} frames successfully")
            viewer = PointCloudSequenceViewer(
                sid,
                pcd_list,
                o3d_img_list,
                seq_infos[-1][:len(pcd_list)],
                line_set_list,
                save_dir,
            )
            viewer.run()
            print(f"  Visualization complete! Results saved to {save_dir}")
            video_path = os.path.join(save_output_dir, batch_name, f"{seq_name}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            create_video_from_frames(save_dir, video_path, fps=10)
            print(f"  Video created: {video_path}")
        return True

    # 主流程：扫描并逐序列处理
    seq_list = list_test_sequences(TEST_ROOT_DIR)
    if not seq_list:
        exit(0)

    print(f"Discovered {len(seq_list)} sequences under {TEST_ROOT_DIR}")
    total_ok, total_fail = 0, 0
    for batch_name, seq_name, seq_dir, calib_json in seq_list:
        print(f"\nProcessing {batch_name}/{seq_name} ...")
        ok = process_sequence(batch_name, seq_name, seq_dir, calib_json)
        if ok:
            total_ok += 1
        else:
            total_fail += 1
    print(f"\nAll done. Succeeded: {total_ok}, Failed/Skipped: {total_fail}")
