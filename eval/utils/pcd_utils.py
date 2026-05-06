import copy
import numpy as np
from scipy.spatial.distance import cdist
from pyquaternion import Quaternion
from .point_cloud import PointCloud
import torch
from typing import Any, Dict, List, Optional, Tuple
from .bounding_box import BoundingBox
def vis(pcd, box, box2=None):
    """
    可视化点云与 BoundingBox，并保存为 JPEG（两列）：
    - 左：整体点云 + bbox（bbox 用线框表示）
    - 右：仅 bbox 内的点云（放大且以 bbox 中心为中心）

    约定：生成文件名 'pcd_vis_<timestamp>.jpg' 并保存在当前工作目录。
    参数:
        pcd: PointCloud 或 可转为 numpy (3,N) 的对象
        box: BoundingBox 对象
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import time

    # 尝试从 PointCloud 中取点，否则直接用传入数组
    try:
        pts = pcd.points.copy()  # shape (3, N)
    except Exception:
        arr = np.asarray(pcd)
        if arr.ndim == 2 and arr.shape[1] == 3:
            pts = arr.T
        elif arr.ndim == 2 and arr.shape[0] == 3:
            pts = arr
        else:
            raise ValueError('pcd must be PointCloud or Nx3 / 3xN array')

    if pts.size == 0:
        print('[vis] empty point cloud, skipping visualization')
        return

    # 转为 N x 3
    pts_n = pts.T

    # 计算点云主分布中心（使用中位数更鲁棒）和尺度（95% 分位范围）
    center = np.median(pts_n, axis=0)
    low = np.percentile(pts_n, 2.5, axis=0)
    high = np.percentile(pts_n, 97.5, axis=0)
    span = high - low
    # 防止太小
    span[span < 1e-3] = 1e-3

    # bbox corners (primary)
    try:
        corners = box.corners()  # (3,8)
    except Exception:
        # 如果不是 BoundingBox，则尝试把 box 转换为 array
        corners = np.asarray(box)
        if corners.shape == (8, 3):
            corners = corners.T

    # 计算哪些点在 bbox 内
    try:
        # get_pcd_in_box_mask expects PointCloud with shape (3, N)
        mask = get_pcd_in_box_mask(PointCloud(pts), box, offset=0, scale=1.0)
    except Exception:
        # 退化为基于 axis-aligned 包围盒判断，使用 pts_n (N,3) 以匹配布尔索引
        cmin = np.min(corners, axis=1)
        cmax = np.max(corners, axis=1)
        mask = np.all((pts_n >= cmin.reshape(1, 3)) & (pts_n <= cmax.reshape(1, 3)), axis=1)

    in_pts = pts_n[mask.astype(bool)] if mask is not None else np.zeros((0, 3))

    # secondary box (optional): compute mask2 and corners2
    mask2 = None
    corners2 = None
    if box2 is not None:
        try:
            corners2 = box2.corners()
        except Exception:
            arr2 = np.asarray(box2)
            if arr2.shape == (8, 3):
                corners2 = arr2.T
        if corners2 is not None:
            try:
                mask2 = get_pcd_in_box_mask(PointCloud(pts), box2, offset=0, scale=1.0)
            except Exception:
                cmin2 = np.min(corners2, axis=1)
                cmax2 = np.max(corners2, axis=1)
                mask2 = np.all((pts_n >= cmin2.reshape(1, 3)) & (pts_n <= cmax2.reshape(1, 3)), axis=1)
    in_pts2 = pts_n[mask2.astype(bool)] if mask2 is not None else np.zeros((0, 3))

    # 可视化：两列子图
    fig = plt.figure(figsize=(12, 6))

    # 左：整体点云 + bbox
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pts_n[:, 0], pts_n[:, 1], pts_n[:, 2], s=1, c='gray', alpha=0.6)
    # 绘制 bbox 边框线
    lines = [(0, 1), (1, 5), (5, 4), (4, 0),
             (3, 2), (2, 6), (6, 7), (7, 3),
             (0, 3), (1, 2), (5, 6), (4, 7)]
    try:
        c = corners.T.reshape(8, 3)
        for i, j in lines:
            xs = [c[i, 0], c[j, 0]]
            ys = [c[i, 1], c[j, 1]]
            zs = [c[i, 2], c[j, 2]]
            ax1.plot(xs, ys, zs, c='r')
    except Exception:
        pass

    # 如果提供第二个 bbox，绘制为绿色并显示计数
    if corners2 is not None:
        try:
            c2 = corners2.T.reshape(8, 3)
            for i, j in lines:
                xs = [c2[i, 0], c2[j, 0]]
                ys = [c2[i, 1], c2[j, 1]]
                zs = [c2[i, 2], c2[j, 2]]
                ax1.plot(xs, ys, zs, c='g')
        except Exception:
            pass

    # 以点云主分布为中心和尺度设置视窗（满足用户要求 ①）
    lim_x = (center[0] - 1.2 * span[0], center[0] + 1.2 * span[0])
    lim_y = (center[1] - 1.2 * span[1], center[1] + 1.2 * span[1])
    lim_z = (center[2] - 1.2 * span[2], center[2] + 1.2 * span[2])
    ax1.set_xlim(lim_x)
    ax1.set_ylim(lim_y)
    ax1.set_zlim(lim_z)
    # 在左图显示两个 bbox 内点数
    count1 = int(in_pts.shape[0])
    count2 = int(in_pts2.shape[0]) if box2 is not None else None
    title = f'PointCloud + BBox | in-box pts: {count1}'
    if count2 is not None:
        title += f' | box2 pts: {count2}'
    ax1.set_title(title)

    # 右：bbox 内点云（放大并以 bbox 中心为中心）
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if in_pts.size > 0:
        ax2.scatter(in_pts[:, 0], in_pts[:, 1], in_pts[:, 2], s=2, c='b', alpha=0.8)
    else:
        ax2.text(0.5, 0.5, 0.5, 'No points in bbox', horizontalalignment='center')

    # bbox center and span
    try:
        bbox_center = np.array(box.center).reshape(3)
        bbox_span = np.max(corners, axis=1) - np.min(corners, axis=1)
    except Exception:
        bbox_center = center
        bbox_span = span

    # 放大系数，确保视野稍大于 bbox
    scale_factor = 1.25
    bx = (bbox_center[0] - bbox_span[0] * scale_factor / 2, bbox_center[0] + bbox_span[0] * scale_factor / 2)
    by = (bbox_center[1] - bbox_span[1] * scale_factor / 2, bbox_center[1] + bbox_span[1] * scale_factor / 2)
    bz = (bbox_center[2] - bbox_span[2] * scale_factor / 2, bbox_center[2] + bbox_span[2] * scale_factor / 2)
    # 若 bbox 某维度过小，用 span 补偿
    if bbox_span[0] < 1e-3:
        bx = (bbox_center[0] - span[0] * 0.5, bbox_center[0] + span[0] * 0.5)
    if bbox_span[1] < 1e-3:
        by = (bbox_center[1] - span[1] * 0.5, bbox_center[1] + span[1] * 0.5)
    if bbox_span[2] < 1e-3:
        bz = (bbox_center[2] - span[2] * 0.5, bbox_center[2] + span[2] * 0.5)

    ax2.set_xlim(bx)
    ax2.set_ylim(by)
    ax2.set_zlim(bz)
    # 绘制 bbox 线框
    try:
        for i, j in lines:
            xs = [c[i, 0], c[j, 0]]
            ys = [c[i, 1], c[j, 1]]
            zs = [c[i, 2], c[j, 2]]
            ax2.plot(xs, ys, zs, c='r')
    except Exception:
        pass

    # 若提供第二 bbox，显示其在右图中的点（同样绘制为绿色轮廓）
    if corners2 is not None:
        try:
            for i, j in lines:
                xs = [c2[i, 0], c2[j, 0]]
                ys = [c2[i, 1], c2[j, 1]]
                zs = [c2[i, 2], c2[j, 2]]
                ax2.plot(xs, ys, zs, c='g')
        except Exception:
            pass

    ax2_title = f'Points inside BBox (zoom) | count: {count1}'
    if count2 is not None:
        ax2_title += f' | box2: {count2}'
    ax2.set_title(ax2_title)

    plt.tight_layout()
    fname = f'pcd_vis_{int(time.time()*1000)}.jpg'
    try:
        fig.savefig(fname, dpi=150)
        print(f'[vis] saved: {fname}')
    except Exception as e:
        print('[vis] failed to save image:', e)
    plt.close(fig)
def roi_crop_3d(pcd, bbox, target_n) -> torch.Tensor:
        """
        裁剪并重采样点云到目标框附近，返回 (N, 3)。
        """
        B, N, _ = pcd.shape
        device = pcd.device
        def _bbox_from_contour(bbox):
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

            cen = to_vec3((bbox[0], bbox[1], bbox[2]))
            siz = to_vec3((bbox[3], bbox[4], bbox[5]))
            rot = to_vec3((bbox[6], bbox[7], bbox[8]))

            L, W, H = float(siz[0]), float(siz[1]), float(siz[2])
            size_wlh = [W, L, H]

            yaw = float(rot[2])
            orientation = Quaternion(axis=[0, 0, 1], radians=yaw)
            bbox = BoundingBox(center=[float(cen[0]), float(cen[1]), float(cen[2])],
                            size=size_wlh, orientation=orientation)
            return bbox
        # bbox -> BoundingBox
        bbox = [_bbox_from_contour(b) for b in bbox.cpu().numpy()]
        # pcd -> PointCloud
        pcd = [PointCloud(p.T) for p in pcd.cpu().numpy()]
        # return tensor
        idx_out = []
        for b in range(B):
            bbox_b = bbox[b]
            pcd_b = pcd[b]
            for scale in (1.25, 1.75, 2.25):
                idx = np.array([], dtype=np.int64)
                try:
                    # pcd_crop = crop_and_center_pcd(
                    #     pcd, bbox
                    # )
                    mask = get_pcd_in_box_mask(pcd_b, bbox_b, scale=scale)
                    if mask is None:
                        continue
                    idx = np.where(mask.astype(bool))[0]
                    if idx.size > 0:
                        pts4 = pcd_b.points[:, idx]
                        pcd_crop = PointCloud(pts4)
                        pcd_crop, idx = resample_pcd(pcd_crop, target_n, return_idx=True, is_training=True)
                        # return torch.as_tensor(pcd_crop.points[:3, :].T, dtype=torch.float32)  # (N,3)
                        break
                except Exception:
                    continue
            # center = np.array(bbox_b.center, dtype=np.float32).reshape(3, 1)
            # jitter = (np.random.rand(3, target_n).astype(np.float32) - 0.5) * 0.02
            # pts3 = (center + jitter).T  # (N,3)
            # if idx.size == 0:
            #     idx = np.random.choice(N, 1, replace=True)
            #     pts4 = pcd_b.points[:, idx]
            #     pcd_crop = PointCloud(pts4)
            #     pcd_crop, idx = resample_pcd(pcd_crop, target_n, return_idx=True, is_training=True)
            
            # 可视化pcd和bbox 并保存到aaaa
            import open3d as o3d 
            # vis(pcd_b, bbox_b)
            if idx.size == 0:
                # -1
                idx = np.array([-1]*target_n, dtype=np.int64)
            idx_out.append(torch.from_numpy(idx))
        idx_out = torch.stack(idx_out, dim=0).to(device)  # [B, target_n]
        return idx_out # [B, target_n]
 
def resample_pcd(pcd, n_sample, is_training=True, return_idx=False):
    # random sampling from points
    pcd = pcd.points.T
    num_points = pcd.shape[0]
    new_pts_idx = None
    rng = np.random if is_training else np.random.default_rng(1)
    if num_points >= 1:
        if num_points < n_sample:
            new_pts_idx = rng.choice(
                num_points, size=n_sample-num_points, replace=True)
            idx = np.arange(num_points)
            rng.shuffle(idx)
            new_pts_idx = np.concatenate(
                [idx, new_pts_idx], axis=0)
        elif num_points > n_sample:
            new_pts_idx = rng.choice(
                num_points, size=n_sample, replace=False)
        else:
            new_pts_idx = np.arange(num_points)
            rng.shuffle(new_pts_idx)
    if new_pts_idx is not None:
        pcd = pcd[new_pts_idx, :].copy()
    else:
        pcd = np.zeros((n_sample, 3), dtype='float32')
    pcd = PointCloud(pcd.T)
    if return_idx:
        return pcd, new_pts_idx
    else:
        return pcd


def crop_pcd_axis_aligned(pcd, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the box in the axis-aligned manner
    """
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = pcd.points[0, :] < maxi[0]
    x_filt_min = pcd.points[0, :] > mini[0]
    y_filt_max = pcd.points[1, :] < maxi[1]
    y_filt_min = pcd.points[1, :] > mini[1]
    z_filt_max = pcd.points[2, :] < maxi[2]
    z_filt_min = pcd.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_pcd = PointCloud(pcd.points[:, close].copy())
    if return_mask:
        return new_pcd, close
    else:
        return new_pcd


def crop_pcd_oriented(pcd, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the exact box.
    slower than 'crop_pc_axis_aligned' but more accurate
    """

    box_tmp = copy.deepcopy(box)
    new_pcd = PointCloud(pcd.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_pcd.translate(trans)
    box_tmp.translate(trans)
    new_pcd.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_pcd.points[0, :] < maxi[0]
    x_filt_min = new_pcd.points[0, :] > mini[0]
    y_filt_max = new_pcd.points[1, :] < maxi[1]
    y_filt_min = new_pcd.points[1, :] > mini[1]
    z_filt_max = new_pcd.points[2, :] < maxi[2]
    z_filt_min = new_pcd.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_pcd = PointCloud(new_pcd.points[:, close])

    # transform back to the original coordinate system
    new_pcd.rotate(np.transpose(rot_mat))
    new_pcd.translate(-trans)
    if return_mask:
        return new_pcd, close
    else:
        return new_pcd


def get_offset_box(box, offset, use_z=True, offset_max=[2.0, 2.0, 1.0], degree=True, is_training=True):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)
    if len(offset) == 3:
        use_z = False
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], degrees=offset[2]) if degree else Quaternion(axis=[0, 0, 1], radians=offset[2]))
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], degrees=offset[3]) if degree else Quaternion(axis=[0, 0, 1], radians=offset[3]))
    if is_training:
        if np.abs(offset[0]) > min(new_box.wlh[0], offset_max[0]):
            offset[0] = np.random.uniform(
                0, min(new_box.wlh[0], offset_max[0])) * np.sign(offset[0])
        if np.abs(offset[1]) > min(new_box.wlh[1], offset_max[1]):
            offset[1] = np.random.uniform(
                0, min(new_box.wlh[1], offset_max[1])) * np.sign(offset[1])
        if use_z and np.abs(offset[2]) > min(new_box.wlh[2], offset_max[2]):
            offset[2] = np.random.uniform(
                0, min(new_box.wlh[2], offset_max[2])) * np.sign(offset[2])
    if use_z:
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def crop_and_center_pcd(pcd, box, offset=0, scale=1.0, offset2=0, normalize=False, return_box=False):
    """
    crop and center the pc using the given box
    """
    new_pcd = crop_pcd_axis_aligned(
        pcd, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    new_pcd.translate(trans)
    new_box.translate(trans)
    new_pcd.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # print('HERE!', new_box, offset+offset2, scale)
    new_pcd = crop_pcd_axis_aligned(
        new_pcd, new_box, offset=offset+offset2, scale=scale)
    # print('HERE!', new_pcd.points.T)
    if normalize:
        new_pcd.normalize(box.wlh)
    if return_box:
        return new_pcd, new_box
    else:
        return new_pcd


def merge_template_pcds(pcds, boxes, offset=0, offset2=0, scale=1.0, normalize=False, return_box=False):
    if len(pcds) == 0:
        return PointCloud(np.ones((3, 0)))
    new_pcd = [np.ones((pcds[0].points.shape[0], 0), dtype='float64')]
    for pcd, box in zip(pcds, boxes):
        cropped_pcd, new_box = crop_and_center_pcd(
            pcd, box, offset=offset, offset2=offset2, scale=scale, normalize=normalize, return_box=True)
        # try:
        if cropped_pcd.nbr_points() > 0:
            new_pcd.append(cropped_pcd.points)

    new_pcd = PointCloud(np.concatenate(new_pcd, axis=1))
    if return_box:
        return new_pcd, new_box
    else:
        return new_pcd


def get_point_to_box_distance(pcd, box):
    """
    generate the BoxCloud for the given pc and box
    :param pc: Pointcloud object or numpy array
    :param box:
    :return:
    """
    if isinstance(pcd, PointCloud):
        points = pcd.points.T.copy()  # N,3
    else:
        points = pcd.copy()  # N,3
        assert points.shape[1] == 3
    box_corners = box.corners()  # 3,8
    box_centers = box.center.reshape(-1, 1)  # 3,1
    box_points = np.concatenate([box_centers, box_corners], axis=1)  # 3,9
    p2b_dist = cdist(points, box_points.T)  # N,9
    return p2b_dist


def get_pcd_in_box_mask(pcd, box, offset=0, scale=1.0):
    """check which points of PC are inside the box"""
    box_tmp = copy.deepcopy(box)
    new_pcd = PointCloud(pcd.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_pcd.translate(trans)
    box_tmp.translate(trans)
    new_pcd.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_pcd.points[0, :] < maxi[0]
    x_filt_min = new_pcd.points[0, :] > mini[0]
    y_filt_max = new_pcd.points[1, :] < maxi[1]
    y_filt_min = new_pcd.points[1, :] > mini[1]
    z_filt_max = new_pcd.points[2, :] < maxi[2]
    z_filt_min = new_pcd.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    assert close.shape[0] == new_pcd.points.shape[1]

    return close


def transform_box(box, ref_box):
    new_box = copy.deepcopy(box)
    new_box.translate(-ref_box.center)
    new_box.rotate(Quaternion(matrix=ref_box.rotation_matrix.T))
    return new_box
