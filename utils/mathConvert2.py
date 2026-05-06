# -*- coding: utf-8 -*-
# @Time    : 2024/6/7
# @Author  : Yifan JIAO
# @Project : SOT_Dataset
# @File    : mathConvert.py

import numpy as np
from transforms3d.euler import euler2mat
import torch
import torch.nn.functional as F
def compute_3d_box_vertices_withlist(bbox):
    """
    bbox: [x, y, z, l, w, h, roll, pitch, yaw]
    return: [8, 3]
    """
    if bbox.ndim == 1:
        cen = bbox[0:3]
        siz = bbox[3:6]
        rot = bbox[6:9]
    else:
        cen = bbox[...,0:3]
        siz = bbox[...,3:6]
        rot = bbox[...,6:9]
    vertices = compute_3d_box_vertices(cen, siz, rot)  # [8,3]
    return vertices

def compute_3d_box_vertices(center, dim, rotation):
    """
    计算 3D 盒子的 8 个角点，严格按照:
      - 坐标系: x=前进, y=右, z=上
      - 尺寸:   dim=[length(x), width(y), height(z)]
      - 旋转:   rotation=[roll(x), pitch(y), yaw(z)] (弧度)，内禀 rxyz
    支持 numpy 或 torch 输入；返回与输入类型一致（torch 优先）。
    """
    import torch

    # 是否使用 torch
    use_torch = isinstance(center, torch.Tensor) or isinstance(dim, torch.Tensor) or isinstance(rotation, torch.Tensor)

    if not use_torch:
        # numpy 路径（与 transforms3d 一致）
        if center is None or dim is None or rotation is None:
            return None
        cx, cy, cz = np.asarray(center, dtype=np.float32)
        length, width, height = np.asarray(dim, dtype=np.float32)

        local_vertices = np.array([
            [ length/2,  width/2,  height/2],
            [ length/2, -width/2,  height/2],
            [-length/2, -width/2,  height/2],
            [-length/2,  width/2,  height/2],
            [ length/2,  width/2, -height/2],
            [ length/2, -width/2, -height/2],
            [-length/2, -width/2, -height/2],
            [-length/2,  width/2, -height/2],
        ], dtype=np.float32)

        rot = np.asarray(rotation, dtype=np.float32)
        if rot.size == 3:
            R = euler2mat(rot[0], rot[1], rot[2], 'rxyz')  # 内禀 rxyz
        elif rot.size == 9:
            R = rot.reshape(3, 3)
        else:
            raise ValueError('rotation 必须是 [3] 或 [3x3]')

        vertices = local_vertices @ R.T + np.array([cx, cy, cz], dtype=np.float32)
        idx = np.array([3, 2, 6, 7, 0, 1, 5, 4])  # 变换顶点顺序
        # 变换为 0: 后上左，1：后上右，2：后下右，3：后下左，4：前上左，5：前上右，6：前下右，7：前下左
        vertices = vertices[idx]
        return vertices.astype(np.float32)

    # torch 路径（支持批量）
    c = center if isinstance(center, torch.Tensor) else torch.tensor(center, dtype=torch.float32)
    d = dim    if isinstance(dim, torch.Tensor)    else torch.tensor(dim,    dtype=torch.float32)
    r = rotation if isinstance(rotation, torch.Tensor) else torch.tensor(rotation, dtype=torch.float32)

    c = c.float(); d = d.float(); r = r.float()
    orig_shape = tuple(c.shape[:-1])
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    l,  w,  h  = d[..., 0], d[..., 1], d[..., 2]
    base = torch.tensor(
        [[ 1,  1,  1],
         [ 1, -1,  1],
         [-1, -1,  1],
         [-1,  1,  1],
         [ 1,  1, -1],
         [ 1, -1, -1],
         [-1, -1, -1],
         [-1,  1, -1]], dtype=torch.float32, device=cx.device)
    local = 0.5 * torch.stack([l, w, h], dim=-1).unsqueeze(-2) * base  # [...,8,3]

    if r.shape[-1] == 3:
        roll, pitch, yaw = r[..., 0], r[..., 1], r[..., 2]
        cr, sr = torch.cos(roll),  torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy_,sy_ = torch.cos(yaw),  torch.sin(yaw)

        Rx = torch.stack([
            torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)], dim=-1),
            torch.stack([torch.zeros_like(cr), cr, -sr], dim=-1),
            torch.stack([torch.zeros_like(cr), sr,  cr], dim=-1),
        ], dim=-2)
        Ry = torch.stack([
            torch.stack([cp, torch.zeros_like(cp), sp], dim=-1),
            torch.stack([torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)], dim=-1),
            torch.stack([-sp, torch.zeros_like(cp), cp], dim=-1),
        ], dim=-2)
        Rz = torch.stack([
            torch.stack([cy_, -sy_, torch.zeros_like(cy_)], dim=-1),
            torch.stack([sy_,  cy_, torch.zeros_like(cy_)], dim=-1),
            torch.stack([torch.zeros_like(cy_), torch.zeros_like(cy_), torch.ones_like(cy_)], dim=-1),
        ], dim=-2)
        # 内禀 rxyz 等价于 R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
    elif r.shape[-2:] == (3, 3):
        R = r
    else:
        raise ValueError('rotation 必须是 [...,3] 或 [...,3,3]')

    rotated = torch.matmul(local, R.transpose(-1, -2))               # [...,8,3]
    center_exp = torch.stack([cx, cy, cz], dim=-1).unsqueeze(-2)      # [...,1,3]
    corners = rotated + center_exp
    # (4) +---------+. (5)
    #     | ` .     |  ` .
    #     | (0) +---+-----+ (1)
    #     |     |   |     |
    # (7) +-----+---+. (6)|
    #     ` .   |     ` . |
    #     (3) ` +---------+ (2)
    # 当前为 0：前上左，1：前上右，2：后上右，3：后上左，4：前下左，5：前下右，6：后下右，7：后下左
    # 变换为 0: 后上左，1：后上右，2：后下右，3：后下左，4：前上左，5：前上右，6：前下右，7：前下左
    idx = torch.tensor([3, 2, 6, 7, 0, 1, 5, 4], dtype=torch.long, device=cx.device)
    corners = corners[..., idx, :]
    return corners.reshape(*orig_shape, 8, 3)

def compute_rotation_matrix(rotation):
    """
    Compute the rotation matrix from yaw, pitch, and roll angles.

    Args:
        rotation (array-like): A 3-element array representing the rotation angles (yaw, pitch, roll) in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    roll, pitch, yaw = rotation

    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    r_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    r_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    r = r_z @ r_y @ r_x
    # r = np.dot(r_z, np.dot(r_y, r_x))
    return r

def compute_rotation_matrix_from_directions(default_direction, target_direction):
    """计算从默认方向到目标方向的旋转矩阵"""
    default_direction = default_direction / np.linalg.norm(default_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)
    v = np.cross(default_direction, target_direction)
    c = np.dot(default_direction, target_direction)
    k = 1 / (1 + c)
    R = np.array([
        [v[0] * v[0] * k + c, v[0] * v[1] * k - v[2], v[0] * v[2] * k + v[1]],
        [v[1] * v[0] * k + v[2], v[1] * v[1] * k + c, v[1] * v[2] * k - v[0]],
        [v[2] * v[0] * k - v[1], v[2] * v[1] * k + v[0], v[2] * v[2] * k + c]
    ])
    return R

def project_3d_to_2d(points_3d, intrinsic, extrinsic):
    """
    批量 3D -> 2D 投影（Torch）
    参数:
        points_3d: [..., N, 3]，世界坐标
        intrinsic: [..., 3, 3] 或 [3,3]
        extrinsic: [..., 4, 4]/[3,4] 或 [4,4]/[3,4]，将世界坐标变换到相机坐标
    返回:
        [..., N, 2]，像素坐标 (u,v)
    """
    def to_tensor(x, device=None, dtype=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        assert torch.is_tensor(x), "inputs must be numpy or torch tensors"
        if device is not None:
            x = x.to(device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x

    pts = to_tensor(points_3d)
    device = pts.device
    dtype = pts.dtype if pts.dtype.is_floating_point else torch.float32
    pts = pts.to(device=device, dtype=dtype)

    K = to_tensor(intrinsic, device=device, dtype=dtype)
    E = to_tensor(extrinsic, device=device, dtype=dtype)

    # 归一化形状
    orig_shape = pts.shape[:-2]  # 可能为空
    N = pts.shape[-2]
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)  # [1,N,3]
    B = int(np.prod(pts.shape[:-2])) if pts.dim() > 3 else pts.shape[0]
    pts = pts.reshape(-1, N, 3)  # [B,N,3]

    # 处理 extrinsic，允许 [3,4] / [4,4] / [B,3,4] / [B,4,4]
    if E.dim() == 2:
        if E.shape == (3, 4):
            bottom = torch.tensor([[0, 0, 0, 1]], device=device, dtype=dtype)
            E = torch.cat([E, bottom], dim=0)
        if E.shape == (4, 4):
            E = E.unsqueeze(0).expand(B, 4, 4)
        else:
            raise ValueError("extrinsic 必须为 [...,4,4] 或 [...,3,4]")
    elif E.dim() == 3:
        if E.shape[-2:] == (3, 4):
            bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).view(1, 1, 4).expand(E.shape[0], -1, -1)
            E = torch.cat([E, bottom], dim=-2)  # [*,4,4]
        elif E.shape[-2:] != (4, 4):
            raise ValueError("extrinsic 尺寸应为 (...,4,4) 或 (...,3,4)")
        if E.shape[0] == 1 and B > 1:
            E = E.expand(B, -1, -1)
        elif E.shape[0] != B:
            E = E.reshape(-1, 4, 4)
            if E.shape[0] != B:
                raise ValueError(f"extrinsic batch={E.shape[0]} 与 points batch={B} 不匹配")
    else:
        raise ValueError("extrinsic 维度错误")

    # 处理 intrinsic，允许 [3,3] / [B,3,3]
    if K.dim() == 2 and K.shape == (3, 3):
        K = K.unsqueeze(0).expand(B, 3, 3)
    elif K.dim() == 3 and K.shape[-2:] == (3, 3):
        if K.shape[0] == 1 and B > 1:
            K = K.expand(B, -1, -1)
        elif K.shape[0] != B:
            K = K.reshape(-1, 3, 3)
            if K.shape[0] != B:
                raise ValueError(f"intrinsic batch={K.shape[0]} 与 points batch={B} 不匹配")
    else:
        raise ValueError("intrinsic 尺寸应为 [3,3] 或 [B,3,3]")

    # 同次坐标
    ones = torch.ones((B, N, 1), device=device, dtype=dtype)
    pts_h = torch.cat([pts, ones], dim=-1)  # [B,N,4]

    # 世界到相机: [B,4,4] @ [B,N,4]^T -> [B,N,4]
    cam = torch.einsum('bij,bnj->bni', E, pts_h)  # [B,N,4]
    xyz = cam[..., :3]

    # 投影到像素
    uvw = torch.einsum('bij,bnj->bni', K, xyz)  # [B,N,3]
    z = uvw[..., 2:3].clamp(min=1e-6)
    uv = uvw[..., :2] / z  # [B,N,2]

    out = uv
    # 还原前置维度
    out = out.reshape(*orig_shape, N, 2) if len(orig_shape) else out.squeeze(0)
    return out

def point_to_xxyyzz(points_3d):
    """
    points_3d: [..., N, 3]
    return: [..., 6]  x_min, x_max, y_min, y_max, z_min, z_max
    """
    min_xyz, _ = torch.min(points_3d, dim=-2)  # [..., 3]
    max_xyz, _ = torch.max(points_3d, dim=-2)  # [..., 3]
    x_min, y_min, z_min = min_xyz[..., 0], min_xyz[..., 1], min_xyz[..., 2]
    x_max, y_max, z_max = max_xyz[..., 0], max_xyz[..., 1], max_xyz[..., 2]
    bbox = torch.stack([x_min, x_max, y_min, y_max, z_min, z_max], dim=-1)  # [..., 6]
    return bbox

def ROI_3d(input, boxes, output_size):
    """
    3D ROI Align on voxel features using trilinear sampling (grid_sample).

    Args:
        input:  Tensor [B, C, D, H, W], features on a fixed metric grid.
        boxes:  Tensor [M, 7] with columns [batch_idx, x_min, x_max, y_min, y_max, z_min, z_max] in meters.
        output_size: tuple(int, int, int) -> (D_out, H_out, W_out)

    Requirements satisfied:
    - Differentiable: uses torch.grid_sample with trilinear interpolation.
    - Axis mapping: x -> D, y -> H, z -> W.
    - Coordinate scale: boxes are in meters; voxel grid spans a fixed space range
      consistent with the 3D backbone default: x∈[0,10], y∈[-2,2], z∈[-1,1].
    - Semantics similar to torchvision.ops.roi_align: regular sampling in the ROI extents.

    Returns:
        Tensor [M, C, D_out, H_out, W_out]
    """
    import torch
    import torch.nn.functional as F

    assert input.dim() == 5, "input must be [B, C, D, H, W]"
    assert boxes.dim() == 2 and boxes.size(1) == 7, "boxes must be [M, 7]"

    B, C, D, H, W = input.shape
    D_out, H_out, W_out = output_size

    device = input.device
    dtype = input.dtype

    # Fixed metric range used by the 3D backbone (see _3DBackbone defaults and pipeline constants)
    # x in [0,10], y in [-2,2], z in [-1,1]
    x_min, x_max = 0.0, 10.0
    y_min, y_max = -2.0, 2.0
    z_min, z_max = -1.0, 1.0

    # Parse boxes
    b_idx = boxes[:, 0].long().to(device)
    x1, x2 = boxes[:, 1].to(device).to(dtype), boxes[:, 2].to(device).to(dtype)
    y1, y2 = boxes[:, 3].to(device).to(dtype), boxes[:, 4].to(device).to(dtype)
    z1, z2 = boxes[:, 5].to(device).to(dtype), boxes[:, 6].to(device).to(dtype)

    M = boxes.size(0)

    # Build per-axis parametric coordinates t∈[0,1], then linear blend with (min,max)
    # This keeps differentiability w.r.t box endpoints.
    t_d = torch.linspace(0.0, 1.0, steps=D_out, device=device, dtype=dtype)  # along x -> D
    t_h = torch.linspace(0.0, 1.0, steps=H_out, device=device, dtype=dtype)  # along y -> H
    t_w = torch.linspace(0.0, 1.0, steps=W_out, device=device, dtype=dtype)  # along z -> W

    # [M, D_out], [M, H_out], [M, W_out]
    x_vals = x1.unsqueeze(-1) * (1 - t_d) + x2.unsqueeze(-1) * t_d
    y_vals = y1.unsqueeze(-1) * (1 - t_h) + y2.unsqueeze(-1) * t_h
    z_vals = z1.unsqueeze(-1) * (1 - t_w) + z2.unsqueeze(-1) * t_w

    # Normalize to [-1, 1] for grid_sample with align_corners=True
    def norm(v, vmin, vmax):
        denom = max(vmax - vmin, 1e-6)
        return 2.0 * (v - vmin) / denom - 1.0

    x_n = norm(x_vals, x_min, x_max).view(M, D_out, 1, 1)  # will map to grid[..., 2] (D axis)
    y_n = norm(y_vals, y_min, y_max).view(M, 1, H_out, 1)  # will map to grid[..., 1] (H axis)
    z_n = norm(z_vals, z_min, z_max).view(M, 1, 1, W_out)  # will map to grid[..., 0] (W axis)

    # Compose grid [M, D_out, H_out, W_out, 3] in (x,y,z) order expected by grid_sample
    # where x->W (world z), y->H (world y), z->D (world x)
    grid = torch.empty((M, D_out, H_out, W_out, 3), device=device, dtype=dtype)
    grid[..., 0] = z_n.expand(-1, D_out, H_out, W_out)  # x (W axis) from world z
    grid[..., 1] = y_n.expand(-1, D_out, H_out, W_out)  # y (H axis) from world y
    grid[..., 2] = x_n.expand(-1, D_out, H_out, W_out)  # z (D axis) from world x

    # Gather feature volumes per ROI batch index -> [M, C, D, H, W]
    feat = torch.index_select(input, dim=0, index=b_idx)

    # Sample with trilinear interpolation; zeros outside range
    out = F.grid_sample(
        feat,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )  # [M, C, D_out, H_out, W_out]

    return out

