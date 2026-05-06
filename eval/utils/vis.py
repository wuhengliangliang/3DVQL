# -*- coding: utf-8 -*-
# @Time    : 2024/4/8
# @Author  : Yifan JIAO
# @Project : SOT_Dataset
# @File    : utils.py

import numpy as np
import pandas as pd
import struct
import open3d as o3d
import shutil
import cv2
import os
import json
from PIL import Image
import subprocess


def read_bin_pc_xyz(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def read_bin_pc_xyzr(file_path: str):
    # 假设二进制点云数据格式为：[x, y, z, r]，每个点占16字节
    # [fb94 bfbf effe 9cba e696 1c3f 00a0 a844] 是[x, y, z, r]的16进制表示，np.float32类型
    # 每两个16进制数代表一个变量，即xyzr分别为fb94bfbf, effe9cba, e6961c3f, 00a0a844
    # 每个数由4个字节组成，长32位，故1个点的长度为16字节
    points = []
    with open(file_path, 'rb') as f:
        while True:
            bytes_ = f.read(16)  # 读取16字节的数，也就是一个点的数据
            if not bytes_:
                break
            x, y, z, r = struct.unpack('ffff', bytes_)  # f代表float32，即读取4个float32类型的数据
            points.append([x, y, z, r])
    return np.array(points)


def bin_to_pcd(file_path: str):
    size_float = 4
    list_pcd = []
    with open(file_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def convert_16uc1_to_cv2(img_bytes, width, height):
    # 将二进制bytes转换为numpy数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint16)

    # 将数组reshape为图像的形状
    img_array = img_array.reshape((height, width))

    # 将图像数组转换为8位灰度图像
    img_uint8 = cv2.convertScaleAbs(img_array, alpha=(255.0 / 65535.0))
    # 返回转换后的图像
    # return img_uint8
    return img_array


def get_file_names_without_extension(folder_path):
    # 获取目标文件夹下的所有文件名称
    file_names = os.listdir(folder_path)
    # 去除文件后缀并排序
    file_names_without_extension = sorted([os.path.splitext(file)[0] for file in file_names])
    return file_names_without_extension


class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = data[1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def in_camera_coordinate(self, is_homogenous=False):
        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3D bounding box vertices [3, 8]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]

        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return points_3d


# =========================================================
# Projections
# =========================================================
def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_cam2_to_velo(calib):
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    # inverse rigid transformation
    velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

    proj_mat = P_cam_ref2velo @ R_ref2rect_inv
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def project_camera_to_lidar(points, proj_mat):
    """
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]

    Returns:
        points in lidar coordinate:     [3, npoints]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    return points[:3, :]


def map_box_to_image(box, proj_mat):
    """
    Projects 3D bounding box into the image plane.
    Args:
        box (Box3D)
        proj_mat: projection matrix
    """
    # box in camera coordinate
    points_3d = box.in_camera_coordinate()

    # project the 3d bounding box into the image plane
    points_2d = project_to_image(points_3d, proj_mat)

    return points_2d


# =========================================================
# Utils
# =========================================================
def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


# =========================================================
# Drawing tool
# =========================================================
def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    qs = qs.astype(np.int32).transpose()
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    return image

def closest_timestamp(target, timestamps, threshold=10):
    """
    在timestamps列表中找到与target最接近的时间戳，且差距不超过阈值
    """
    closest = min(timestamps, key=lambda x: abs(int(x) - int(target)))
    if abs(int(closest) - int(target)) <= threshold:
        return closest
    return None


def align_files(img_folder, pcd_folder):
    """
    对齐图片和点云文件夹中的文件，并重命名为序号格式（例如0001, 0002, ...）
    """
    # 获取图片和点云文件的列表
    image_files = sorted(os.listdir(img_folder))
    pcd_files = sorted(os.listdir(pcd_folder))

    # 仅保留文件名中的时间戳部分（假设文件名是时间戳）
    image_timestamps = [os.path.splitext(file)[0] for file in image_files]
    pcd_timestamps = [os.path.splitext(file)[0] for file in pcd_files]

    # 找到第一帧图片对应的最近点云文件
    for ts in image_timestamps:
        closest_pc_ts = closest_timestamp(ts, pcd_timestamps)
        if closest_pc_ts is not None:
            break
        image_files.pop(0)
    closest_pc_file = [file for file in pcd_files if os.path.splitext(file)[0] == closest_pc_ts][0]

    # 确定第一帧点云文件的索引
    first_pc_index = pcd_files.index(closest_pc_file)

    # 重命名所有图片文件
    for i, image_file in enumerate(image_files):
        new_name = f"{i + 1:04d}" + os.path.splitext(image_file)[1]
        src_path = os.path.join(img_folder, image_file)
        dst_path = os.path.join(img_folder, new_name)
        shutil.move(src_path, dst_path)
        print(f"Renamed image '{src_path}' to '{dst_path}'")

    # 重命名所有点云文件
    for i in range(len(pcd_files)):
        pc_index = (first_pc_index + i) % len(pcd_files)
        new_name = f"{i + 1:04d}" + os.path.splitext(pcd_files[pc_index])[1]
        src_path = os.path.join(pcd_folder, pcd_files[pc_index])
        dst_path = os.path.join(pcd_folder, new_name)
        shutil.move(src_path, dst_path)
        print(f"Renamed point cloud '{src_path}' to '{dst_path}'")


def del_hidden_file(dir_path, prefix="._*"):
    """
    delete hidden file in OS X
    :param dir_path:
    :param prefix: "._" or "."
    :return:
    """
    try:
        command = ["find", dir_path, "-name", prefix, "-delete"]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印错误信息
        print(f"Error executing find command: {e}")
    except Exception as e:
        # 处理其他可能的错误
        print(f"An error occurred: {e}")


def remove_directory(directory_path):
    try:
        # 执行 `rm -rf` 命令
        subprocess.run(['rm', '-rf', directory_path], check=True)
        print(f"Successfully removed the directory: {directory_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error removing directory {directory_path}: {e}")


def colorize_depth(uint16_img):
    """
    将8位深度图转换为伪彩色图
    :param uint16_img:
    :return:
    """
    # convert 16bit depth to 8-bit depth (65536 -> 256)
    uint16_img_ = uint16_img - uint16_img.min()
    uint16_img_ = uint16_img_ / (uint16_img_.max() - uint16_img_.min())
    uint16_img_ *= 255
    # near blue, far red
    uint16_img_ = 255 - uint16_img_

    im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint16_img_, alpha=1), cv2.COLORMAP_JET)
    # convert to mat png
    # im = Image.fromarray(im_color)
    # save image
    return im_color


def if_has_same_file_num(dir_list, root_path):
    """
    Return whether the number of files in each directory is the same.

    Args:
        dir_list: list of directories
        root_path: root path
    """
    # use set to judge whether the number of files is the same
    nums = set()
    for cur_dir in dir_list:
        cur_dir_path = os.path.join(root_path, cur_dir)
        # remove "._*" files
        del_hidden_file(cur_dir_path, prefix="._*")
        del_hidden_file(cur_dir_path, prefix=".*")

        # 利用集合的唯一性判断文件数量是否相等
        cur_file_num = len(os.listdir(cur_dir_path))
        nums.add(cur_file_num)

    return len(nums) == 1, nums


# def add_text_to_image(image, text, pos=(10, 60), font_scale=2.0, font_color=(255, 0, 0), thickness=3):
#     # Convert to BGR for OpenCV
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     # Add text to the image using OpenCV
#     cv2.putText(image_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
#     # Convert back to RGB for matplotlib
#     return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def add_text_to_image(image, text, pos=None, font_scale=2.0, font_color=(255, 0, 0), thickness=3, line_spacing=1.5):
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if pos is None:
        # 如果没有提供位置，处理换行并居中显示
        lines = text.split('\n')
        text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]
        image_height, image_width = image_bgr.shape[:2]
        total_text_height = int(text_height * len(lines) * line_spacing)
        x = image_width // 2
        y = (image_height - total_text_height) // 2 + text_height

        for i, line in enumerate(lines):
            y_position = y + int(i * text_height * line_spacing)
            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            line_width = text_size[0]
            line_x = x - line_width // 2
            cv2.putText(image_bgr, line, (line_x, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    else:
        # 如果提供了位置，直接在指定位置绘制文本
        cv2.putText(image_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

    # Convert back to RGB for matplotlib
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


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
                # occlusion_value = None
                # out_of_view_value = None
                # for value in class_values:
                #     if value.get('alias') == 'occlusion':
                #         occlusion_value = value.get('value')
                #     elif value.get('alias') == 'out-of-view':
                #         out_of_view_value = value.get('value')

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
                    # 'occlusion': occlusion_value,
                    # 'out-of-view': out_of_view_value,
                    'pointN': point_n,
                    'size3D': size3d,
                    'center3D': center3d,
                    'rotation3D': rotation3d
                }

                frames_info.append(frame_info)

    return frames_info


def parse_label_file_txt(filepath):
    df = pd.read_csv(
        filepath,
        sep=' ',
        names=[
            "center_x", "center_y", "center_z",
            "length", "width", "height",
            "roll", "pitch", "yaw", "occlusion"])
    # 合并新列
    df['size3D'] = df[['length', 'width', 'height']].values.tolist()
    df['center3D'] = df[['center_x', 'center_y', 'center_z']].values.tolist()
    df['rotation3D'] = df[['roll', 'pitch', 'yaw']].values.tolist()
    df['occlusion'] = df['occlusion'].astype(int)
    frames_info = df[['size3D', 'center3D', 'rotation3D', 'occlusion']].to_dict(orient='records')
    return frames_info


def parse_label_file_from_prediction(filepath):
    df = pd.read_csv(
        filepath,
        sep=' ',
        names=[
            "center_x", "center_y", "center_z",
            "length", "width", "height",
            "roll", "pitch", "yaw", "occlusion"])
    # 合并新列
    df['size3D'] = df[['length', 'width', 'height']].values.tolist()
    df['center3D'] = df[['center_x', 'center_y', 'center_z']].values.tolist()
    df['rotation3D'] = df[['roll', 'pitch', 'yaw']].values.tolist()
    frames_info = df[['size3D', 'center3D', 'rotation3D', 'occlusion']].to_dict(orient='records')
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
