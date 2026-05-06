import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchvision import transforms
from einops import rearrange
# 3D/2D 统一的 assign_labels（你上面已经改为同时支持 4/9 维）
from utils.anchor_utils import assign_labels, bbox_overlaps_3d
from dataset import dataset_utils
from torch import Tensor

# 如需 2D ratio（可选，用于某些分支），保留默认表
default_aspect_ratios = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])

# =========================================================
#            9-DoF Soft-OBB IoU（可微，批量）
# =========================================================

def _euler_rxyz_to_R(angles: Tensor) -> Tensor:
    """
    angles: (B,3) = [roll(x), pitch(y), yaw(z)]，rxyz 外旋
    return: (B,3,3) local->world
    """
    rx, ry, rz = angles.unbind(-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx,
        torch.zeros_like(cx), sx,  cx
    ], dim=-1).reshape(-1,3,3)

    Ry = torch.stack([
         cy, torch.zeros_like(cy),  sy,
         torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy, torch.zeros_like(cy),  cy
    ], dim=-1).reshape(-1,3,3)

    Rz = torch.stack([
         cz, -sz, torch.zeros_like(cz),
         sz,  cz, torch.zeros_like(cz),
         torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)
    ], dim=-1).reshape(-1,3,3)

    return Rz @ Ry @ Rx  # rxyz: 先 x 再 y 再 z

@torch.no_grad()
def _pair_sampling_points(c_pred: Tensor, s_pred: Tensor, c_gt: Tensor, s_gt: Tensor,
                          n_points: int = 2048, expand: float = 1.25) -> Tensor:
    """
    在覆盖两盒并集的对齐立方体内均匀采样；不参与梯度。
    returns: (B, N, 3)
    """
    B = c_pred.shape[0]
    r_pred = 0.5 * s_pred.norm(dim=1)  # 外接球半径
    r_gt   = 0.5 * s_gt.norm(dim=1)
    extent = (r_pred + r_gt) * expand                 # (B,)
    center = 0.5 * (c_pred + c_gt)                    # (B,3)
    u = torch.rand(B, n_points, 3, device=c_pred.device) * 2.0 - 1.0
    pts = center[:, None, :] + u * extent[:, None, None]
    return pts

def _occupancy_prob(points: Tensor, center: Tensor, size: Tensor, R: Tensor, k: float) -> Tensor:
    """
    软占据概率：sigma(k*(s/2 - |y|)) 三轴相乘；可导。
    points: (B,N,3) world; center/size: (B,3); R: (B,3,3) local->world
    return: (B,N)
    """
    y = (points - center[:, None, :]) @ R.transpose(1, 2)  # world->local
    margin = size[:, None, :] * 0.5 - y.abs()              # (B,N,3)
    prob_axis = torch.sigmoid(k * margin).clamp_min(1e-6)
    return prob_axis.prod(dim=-1)                          # (B,N)

def _soft_obb_iou_core(c_pred: Tensor, s_pred: Tensor, a_pred: Tensor,
                       c_gt: Tensor,   s_gt: Tensor,   a_gt: Tensor,
                       n_points: int = 2048, k: float = 40.0, chunk: int = 0) -> Tensor:
    """
    9-DoF Soft IoU 可微近似。返回 (B,)
    chunk>0 时对 batch 分块以控显存。
    """
    B = c_pred.shape[0]
    R_pred = _euler_rxyz_to_R(a_pred)
    R_gt   = _euler_rxyz_to_R(a_gt)
    pts = _pair_sampling_points(c_pred, s_pred, c_gt, s_gt, n_points=n_points)  # (B,N,3)

    def occ_in_chunks(center, size, R):
        if chunk and chunk < B:
            outs = []
            for i in range(0, B, chunk):
                j = min(i + chunk, B)
                outs.append(_occupancy_prob(pts[i:j], center[i:j], size[i:j], R[i:j], k))
            return torch.cat(outs, dim=0)
        else:
            return _occupancy_prob(pts, center, size, R, k)

    occ_p = occ_in_chunks(c_pred, s_pred, R_pred)
    occ_g = occ_in_chunks(c_gt,   s_gt,   R_gt)

    inter = (occ_p * occ_g).sum(dim=1)
    Sa    = occ_p.sum(dim=1)
    Sb    = occ_g.sum(dim=1)
    return inter / (Sa + Sb - inter + 1e-8)

def soft_obb_iou_with_symmetry(c_pred: Tensor, s_pred: Tensor, a_pred: Tensor,
                               c_gt: Tensor,   s_gt: Tensor,   a_gt: Tensor,
                               symm_flags: Tensor = None, symm_axis: int = 2,
                               K: int = 60, tau: float = 20.0,
                               n_points: int = 2048, k: float = 40.0, chunk: int = 0) -> Tensor:
    """
    对称时对 pred 绕 symm_axis 旋转 K 个角，用 softmax(tau*iou) 近似 max；可导。
    非对称直接走核心 IoU。
    返回 (B,)
    """
    B = c_pred.shape[0]
    if symm_flags is None or (symm_flags.numel() == 0) or (symm_flags.sum() == 0):
        return _soft_obb_iou_core(c_pred, s_pred, a_pred, c_gt, s_gt, a_gt,
                                  n_points=n_points, k=k, chunk=chunk)

    out = torch.empty(B, device=c_pred.device, dtype=c_pred.dtype)

    # 非对称
    m0 = ~symm_flags
    if m0.any():
        out[m0] = _soft_obb_iou_core(c_pred[m0], s_pred[m0], a_pred[m0],
                                     c_gt[m0],   s_gt[m0],   a_gt[m0],
                                     n_points=n_points, k=k, chunk=chunk)

    # 对称：softmax 聚合
    m1 = symm_flags
    if m1.any():
        angles = torch.linspace(0, 2*torch.pi, steps=K, device=c_pred.device)  # (K,)
        a_rep = a_pred[m1][:, None, :].repeat(1, K, 1)
        a_rep[..., symm_axis] = a_rep[..., symm_axis] + angles[None, :]
        b1 = a_rep.shape[0]

        iou_k = _soft_obb_iou_core(
            c_pred[m1].repeat_interleave(K, 0),
            s_pred[m1].repeat_interleave(K, 0),
            a_rep.reshape(-1, 3),
            c_gt[m1].repeat_interleave(K, 0),
            s_gt[m1].repeat_interleave(K, 0),
            a_gt[m1].repeat_interleave(K, 0),
            n_points=n_points, k=k, chunk=0
        ).reshape(b1, K)

        w = torch.softmax(tau * iou_k, dim=1)
        out[m1] = (w * iou_k).sum(dim=1)

    return out


# =========================================================
#                      你的辅助函数
# =========================================================

def _align_pred_gt_for_giou(pred_btNc, gt_btXc, mask_btN):
    """
    pred_btNc: (B*T, N, C)
    gt_btXc:   (B*T, X, C)  允许 X=1 或 X=N 或 其他（会降成 1）
    mask_btN:  (B*T, N)     Bool/Byte

    返回:
      bbox_p_flat: (B*T*N, C)
      bbox_g_flat: (B*T*N, C)
      mask_flat:   (B*T*N,)
    """
    assert pred_btNc.dim() == 3, f"pred shape wrong: {pred_btNc.shape}"
    assert gt_btXc.dim() == 3, f"gt shape wrong: {gt_btXc.shape}"
    assert mask_btN.dim() == 2, f"mask shape wrong: {mask_btN.shape}"

    BT, N, C = pred_btNc.shape
    BT2, X, C2 = gt_btXc.shape
    assert BT == BT2 and C == C2, f"pred {pred_btNc.shape} vs gt {gt_btXc.shape} mismatch"

    # 规范化 GT 的第二维：优先使用 1，其次 N，其他情况取第一个并扩展
    if X == 1:
        gt_btNc = gt_btXc.expand(BT, N, C)                 # 广播到 N
    elif X == N:
        gt_btNc = gt_btXc                                   # 已对齐
    else:
        gt_btNc = gt_btXc[:, :1, :].expand(BT, N, C)

    bbox_p_flat = pred_btNc.reshape(-1, C)
    bbox_g_flat = gt_btNc.reshape(-1, C)
    mask_flat   = mask_btN.reshape(-1)

    return bbox_p_flat, bbox_g_flat, mask_flat


# =========================================================
#                        损失主流程
# =========================================================

def get_losses_with_anchor(config, preds, gts):
    """
    3D 版（与上面一致）：
      preds:
        center: [b,t,N,3]
        size:   [b,t,N,3]  # half-size
        rot:    [b,t,N,3]
        bbox:   [b,t,N,9]
        prob:   [b,t,N]
        anchor: [1,1,N,9]
    """
    if config.train.use_hnm:
        gts = dataset_utils.replicate_sample_for_hnm(gts)

    pred_center = preds['center']        # [b,t,N,3]
    pred_size   = preds['size']          # [b,t,N,3] (half-size)
    pred_rot    = preds['rot']           # [b,t,N,3]
    pred_bbox   = preds['bbox']          # [b,t,N,9]
    pred_prob   = preds['prob']          # [b,t,N]
    anchor      = preds['anchor']        # [1,1,N,9]
    if 'prob_refine' in preds.keys():
        pred_prob_refine = preds['prob_refine']   # [b,t]

    b, t, N = pred_prob.shape
    device = pred_prob.device

    # GT 取 3D 接口（若缺失则从 2D 回退）
    if 'clip_pcd_bbox' in gts:
        gt_bbox = gts['clip_pcd_bbox']           # [b,t,9]
        if 'center' not in gts.keys():
            gts['center'] = gt_bbox[..., :3]
        if 'size' not in gts.keys():
            gts['size']   = gt_bbox[..., 3:6]
        if 'rot' not in gts.keys():
            gts['rot']    = gt_bbox[..., 6:9]
    else:
        # 回退（不推荐，仅兼容）
        gt_bbox_2d = gts['clip_bbox']             # [b,t,4]
        zeros_3 = torch.zeros_like(gt_bbox_2d[..., :1]).repeat(1,1,3)
        zeros_r = torch.zeros_like(gt_bbox_2d[..., :1]).repeat(1,1,3)
        cxcy = (gt_bbox_2d[..., :2] + gt_bbox_2d[..., 2:]) / 2.0
        gt_bbox = torch.cat([cxcy, zeros_3, zeros_3, zeros_r], dim=-1)

        if 'center' not in gts.keys():
            gts['center'] = gt_bbox[..., :3]
        if 'size' not in gts.keys():
            gts['size']   = gt_bbox[..., 3:6]
        if 'rot' not in gts.keys():
            gts['rot']    = gt_bbox[..., 6:9]

    gt_center = gts['center']                    # [b,t,3]
    gt_size   = gts['size']                      # [b,t,3]
    gt_rot    = gts['rot']                       # [b,t,3]
    gt_prob   = gts['clip_with_bbox']            # [b,t]
    gt_before_query = gts['before_query']        # [b,t]

    # 3D anchors 分配正样本（投影 2D AABB IoU）
    if gt_prob.bool().any():
        assign_label = assign_labels(
            anchor.repeat(b, t, 1, 1),           # [b,t,N,9]
            gt_bbox,                              # [b,t,9]
            iou_threshold=config.model.positive_threshold,
            topk=config.model.positive_topk
        )                                        # [b,t,N]
        positive = torch.logical_and(
            gt_prob.unsqueeze(-1).repeat(1,1,N).bool(),
            assign_label.bool()
        )                                        # [b,t,N]
        positive_flat = rearrange(positive, 'b t N -> (b t N)')
    else:
        positive = torch.zeros(b, t, N, device=device).bool()
        positive_flat = rearrange(positive, 'b t N -> (b t N)')

    # 至少保底一个 True
    if torch.sum(positive_flat.float()).item() == 0:
        positive_flat[:1] = True

    # ---- 回归损失（仅正样本）----
    if torch.sum(positive_flat.float()).item() > 0:
        # center
        pred_center_flat = rearrange(pred_center, 'b t N c -> (b t N) c')
        gt_center_rep = rearrange(gt_center.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_center = F.l1_loss(pred_center_flat[positive_flat], gt_center_rep[positive_flat])

        # size (half-size)
        pred_size_flat = rearrange(pred_size, 'b t N c -> (b t N) c')
        gt_size_rep = rearrange(gt_size.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_size = F.l1_loss(pred_size_flat[positive_flat], gt_size_rep[positive_flat])

        # rot
        pred_rot_flat = rearrange(pred_rot, 'b t N c -> (b t N) c')
        gt_rot_rep = rearrange(gt_rot.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_rot = F.l1_loss(pred_rot_flat[positive_flat], gt_rot_rep[positive_flat])

        # --------- 9D 可微 IoU（Soft-OBB）---------
        pred_bbox_btNc = rearrange(pred_bbox, 'b t N c -> (b t) N c')
        gt_bbox_bt1c   = rearrange(gt_bbox.unsqueeze(2), 'b t N c -> (b t) N c')   # N=1
        BT, Nn, C = pred_bbox_btNc.shape
        gt_bbox_btXc = gt_bbox_bt1c.view(BT, -1, C)

        # mask: [BT,N]
        pos_btN = rearrange(positive_flat, '(b t N) -> (b t) N', b=b, t=t)
        bbox_p_flat, bbox_g_flat, mask_flat = _align_pred_gt_for_giou(
            pred_btNc=pred_bbox_btNc,
            gt_btXc=gt_bbox_btXc,
            mask_btN=pos_btN.bool()
        )

        # 对称标记（可选）
        symm_flags = None
        if 'is_symmetric' in gts:
            symm_flags = gts['is_symmetric'].unsqueeze(2).repeat(1,1,N).reshape(-1)
            symm_flags = symm_flags.bool()

        iou, giou, loss_giou = GiouLoss(
            bbox_p_flat, bbox_g_flat, mask=mask_flat,
            symm_flags=symm_flags,    # 可为 None
            symm_axis=2,              # 你的垂直轴，若为 x 则改 0；为 y 改 1
            n_points=2048, k=40.0,
            K=60, tau=20.0,
            chunk=4096
        )

        # 展平回 [b*t*N,c] 方便后面取 top
        pred_bbox_flat = rearrange(pred_bbox_btNc, '(b t) N c -> (b t N) c', b=b, t=t)

    else:
        pred_bbox_flat = rearrange(pred_bbox, 'b t N c -> (b t N) c')
        loss_center = torch.tensor(0.0, device=device)
        loss_size   = torch.tensor(0.0, device=device)
        loss_rot    = torch.tensor(0.0, device=device)
        iou         = torch.tensor(0.0, device=device)
        giou        = None
        loss_giou   = torch.tensor(0.0, device=device)

    # 发生概率（anchor hit）- focal loss on anchors for valid frames
    pred_prob_vec = rearrange(pred_prob, 'b t N -> (b t N)')
    valid_mask = rearrange(gt_before_query.unsqueeze(2).repeat(1,1,N), 'b t N -> (b t N)')
    loss_prob = focal_loss(
        pred_prob_vec[valid_mask].float(),
        positive_flat[valid_mask].float()
    )

    # refine 概率（可选）
    if 'prob_refine' in preds.keys():
        pred_prob_refine = pred_prob_refine.reshape(-1)
        weight = torch.tensor(config.loss.prob_bce_weight, device=gt_prob.device)
        weight_ = weight[gt_prob[gt_before_query.bool()].long()].reshape(-1)
        criterion = nn.BCEWithLogitsLoss(reduce=False)
        loss_prob_refine = (
            criterion(pred_prob_refine[gt_before_query.reshape(-1).bool()],
                      gt_prob[gt_before_query.bool()]) * weight_
        ).mean()

    # 汇总
    loss = {
        'loss_bbox_center': loss_center,
        'loss_bbox_size':   loss_size,
        'loss_bbox_rot':    loss_rot,
        'loss_bbox_giou':   loss_giou,
        'loss_prob':        loss_prob,
        # weights
        'weight_bbox_center': config.loss.weight_bbox_center,
        'weight_bbox_size':   config.loss.weight_bbox_size,
        'weight_bbox_rot':    config.loss.weight_bbox_rot,
        'weight_bbox_giou':   config.loss.weight_bbox_giou,
        'weight_prob':        config.loss.weight_prob,
        # information
        'iou':  iou.detach() if torch.is_tensor(iou) else torch.tensor(0.0, device=device),
        'giou': (giou if giou is not None else iou).detach() if torch.is_tensor(iou) else torch.tensor(0.0, device=device)
    }
    if 'prob_refine' in preds.keys():
        loss.update({
            'loss_prob_refine':  loss_prob_refine,
            'weight_prob_refine': 1.0
        })

    # 取每帧 top1 预测（按 prob）
    pred_prob_btN = rearrange(pred_prob_vec, '(B N) -> B N', N=N)       # [b*t,N]
    pred_bbox_btNc = rearrange(pred_bbox_flat, '(B N) c -> B N c', N=N) # [b*t,N,9]
    pred_prob_top, top_idx = torch.max(pred_prob_btN, dim=-1)           # [b*t]
    pred_bbox_top = torch.gather(
        pred_bbox_btNc, dim=1,
        index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,9)
    ).squeeze(1)                                                        # [b*t,9]

    pred_top = {
        'bbox': rearrange(pred_bbox_top, '(b t) c -> b t c', b=b, t=t),
        'prob': rearrange(pred_prob_top, '(b t) -> b t', b=b, t=t)
    }
    if 'prob_refine' in preds.keys():
        pred_top = {
            'bbox':        rearrange(pred_bbox_top, '(b t) c -> b t c', b=b, t=t),
            'prob_anchor': rearrange(pred_prob_top, '(b t) -> b t', b=b, t=t),
            'prob':        rearrange(pred_prob_refine, '(b t) -> b t', b=b, t=t)
        }

    return loss, pred_top, gts


def get_losses_head(config, refine_prob, gts, preds_top):
    """
    refine_prob: [b,t]
    依据 preds_top['bbox'] (3D) 与 gt (3D) 生成 pseudo label
    """
    b, t = refine_prob.shape
    device = refine_prob.device

    gt_prob = gts['clip_with_bbox']          # [b,t]
    gt_before_query = gts['before_query']    # [b,t]
    gt_bbox = gts['clip_pcd_bbox']           # [b,t,9]

    gt_prob = gt_prob.reshape(-1)
    gt_before_query = gt_before_query.reshape(-1)
    gt_bbox = gt_bbox.reshape(-1, 9)

    refine_prob = refine_prob.reshape(-1)
    pred_bbox = preds_top['bbox'].reshape(-1, 9)

    # 对称标记（可选）
    symm_flags = None
    if 'is_symmetric' in gts:
        symm_flags = gts['is_symmetric'].reshape(-1).bool()

    iou, giou, _ = GiouLoss(
        pred_bbox, gt_bbox,
        symm_flags=symm_flags,
        symm_axis=2, n_points=2048, k=40.0, K=60, tau=20.0, chunk=4096
    )     # [b*t]

    gt_prob_refine = (iou > config.model.positive_threshold).float()

    weight = torch.tensor(config.loss.prob_bce_weight, device=gt_prob.device)
    weight_ = weight[gt_prob_refine[gt_before_query.bool()].long()]
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss_prob_refine = (
        criterion(refine_prob[gt_before_query.reshape(-1).bool()],
                  gt_prob_refine[gt_before_query.bool()]) * weight_
    ).mean()

    loss = {
        'loss_refine_prob':  loss_prob_refine,
        'weight_refine_prob': 1.0
    }
    return loss, gt_prob_refine.reshape(b, t)


def get_losses(config, preds, gts):
    """
    若不使用 anchor 的路径（保持 3D 对齐）
      preds:
        center [b,t,3], size [b,t,3], rot [b,t,3], bbox [b,t,9], prob [b,t]
      gts:
        clip_pcd_bbox [b,t,9], clip_with_bbox [b,t], before_query [b,t]
    """
    pred_center = rearrange(preds['center'], 'b t c -> (b t) c')
    pred_size   = rearrange(preds['size'],   'b t c -> (b t) c')
    pred_rot    = rearrange(preds['rot'],    'b t c -> (b t) c')
    pred_bbox   = rearrange(preds['bbox'],   'b t c -> (b t) c')
    pred_prob   = preds['prob'].reshape(-1)

    if 'center' not in gts.keys():
        gts['center'] = gts['clip_pcd_bbox'][..., :3]
    if 'size' not in gts.keys():
        gts['size']   = gts['clip_pcd_bbox'][..., 3:6]
    if 'rot' not in gts.keys():
        gts['rot']    = gts['clip_pcd_bbox'][..., 6:9]
    gt_center = rearrange(gts['center'],        'b t c -> (b t) c')
    gt_size   = rearrange(gts['size'],          'b t c -> (b t) c')
    gt_rot    = rearrange(gts['rot'],           'b t c -> (b t) c')
    gt_bbox   = rearrange(gts['clip_pcd_bbox'], 'b t c -> (b t) c')
    gt_prob   = gts['clip_with_bbox'].reshape(-1)
    gt_before_query = gts['before_query'].reshape(-1)

    # bbox 回归损失
    loss_center = F.l1_loss(pred_center[gt_prob.bool()], gt_center[gt_prob.bool()])
    loss_size   = F.l1_loss(pred_size[gt_prob.bool()],   gt_size[gt_prob.bool()])
    loss_rot    = F.l1_loss(pred_rot[gt_prob.bool()],    gt_rot[gt_prob.bool()])

    # 9D Soft-OBB IoU
    symm_flags = None
    if 'is_symmetric' in gts:
        symm_flags = gts['is_symmetric'].reshape(-1).bool()

    iou, giou, loss_giou = GiouLoss(
        pred_bbox, gt_bbox, mask=gt_prob.bool(),
        symm_flags=symm_flags,
        symm_axis=2, n_points=2048, k=40.0, K=60, tau=20.0, chunk=4096
    )

    # prob loss
    weight = torch.tensor(config.loss.prob_bce_weight, device=gt_prob.device)
    weight_ = weight[gt_prob[gt_before_query.bool()].long()].reshape(-1)
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss_prob = (
        criterion(pred_prob[gt_before_query.bool()], gt_prob[gt_before_query.bool()]) * weight_
    ).mean()

    loss = {
        'loss_bbox_center': loss_center,
        'loss_bbox_size':   loss_size,
        'loss_bbox_rot':    loss_rot,
        'loss_bbox_giou':   loss_giou,
        'loss_prob':        loss_prob,
        # weights
        'weight_bbox_center': config.loss.weight_bbox_center,
        'weight_bbox_size':   config.loss.weight_bbox_size,
        'weight_bbox_rot':    config.loss.weight_bbox_rot,
        'weight_bbox_giou':   config.loss.weight_bbox_giou,
        'weight_prob':        config.loss.weight_prob,
        # info
        'iou':  iou.detach(),
        'giou': (giou if giou is not None else iou).detach()
    }
    return loss


def GiouLoss(bbox_p, bbox_g, mask=None,
             # Soft-OBB 相关参数
             symm_flags: torch.Tensor = None,  # (N,) bool
             symm_axis: int = 2,               # 0/1/2
             n_points: int = 2048, k: float = 40.0,
             K: int = 60, tau: float = 20.0,
             chunk: int = 0):
    """
    支持：
      bbox_p: (N,4) 或 (N,9)
      bbox_g: (N,4) 或 (N,9)；也可为 (B,N,9) / (B,N,4)
    2D：标准 GIoU；
    9D：改为 9-DoF 可微 OBB-IoU（Soft-OBB）。GIoU 概念不适于软体积，返回 giou=None。
    """
    device = bbox_p.device
    C_p, C_g = bbox_p.shape[-1], bbox_g.shape[-1]
    assert C_p == C_g, f"bbox_p dim {C_p} != bbox_g dim {C_g}"

    # 展平到 (N, C)
    bbox_p = bbox_p.view(-1, C_p)
    bbox_g = bbox_g.view(-1, C_g)

    # 2D 分支（保持不变）
    if C_p == 4:
        x1p = torch.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1, 1)
        x2p = torch.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1, 1)
        y1p = torch.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1, 1)
        y2p = torch.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1, 1)
        bbox_p = torch.cat([x1p, y1p, x2p, y2p], dim=1)

        area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
        area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])

        x1I = torch.maximum(bbox_p[:, 0], bbox_g[:, 0])
        y1I = torch.maximum(bbox_p[:, 1], bbox_g[:, 1])
        x2I = torch.minimum(bbox_p[:, 2], bbox_g[:, 2])
        y2I = torch.minimum(bbox_p[:, 3], bbox_g[:, 3])
        inter_w = torch.maximum(x2I - x1I, torch.tensor(0.0, device=device))
        inter_h = torch.maximum(y2I - y1I, torch.tensor(0.0, device=device))
        I = inter_w * inter_h

        x1C = torch.minimum(bbox_p[:, 0], bbox_g[:, 0])
        y1C = torch.minimum(bbox_p[:, 1], bbox_g[:, 1])
        x2C = torch.maximum(bbox_p[:, 2], bbox_g[:, 2])
        y2C = torch.maximum(bbox_p[:, 3], bbox_g[:, 3])
        area_c = (x2C - x1C) * (y2C - y1C)

        U = area_p + area_g - I
        iou = I / (U + 1e-6)
        giou = iou - (area_c - U) / (area_c + 1e-6)

        if torch.is_tensor(mask):
            loss_giou = torch.mean(1.0 - giou[mask])
        else:
            loss_giou = torch.mean(1.0 - giou)
        return iou, giou, loss_giou

    # 9D 分支：Soft-OBB IoU（可微）
    elif C_p == 9:
        c_pred, s_pred, a_pred = bbox_p[:, 0:3], bbox_p[:, 3:6], bbox_p[:, 6:9]
        c_gt,   s_gt,   a_gt   = bbox_g[:, 0:3], bbox_g[:, 3:6], bbox_g[:, 6:9]

        if symm_flags is not None:
            symm_flags = symm_flags.reshape(-1).to(dtype=torch.bool, device=bbox_p.device)
            assert symm_flags.numel() == bbox_p.shape[0]

        iou = soft_obb_iou_with_symmetry(
            c_pred, s_pred, a_pred,
            c_gt,   s_gt,   a_gt,
            symm_flags=symm_flags,
            symm_axis=symm_axis,
            K=K, tau=tau,
            n_points=n_points, k=k, chunk=chunk
        )  # (N,)

        if torch.is_tensor(mask):
            loss = torch.mean(1.0 - iou[mask])
        else:
            loss = torch.mean(1.0 - iou)
        giou = None
        return iou, giou, loss

    else:
        raise ValueError(f'Unsupported bbox dim {C_p}, only 4 or 9 are supported.')


def get_bbox_ratio(size, device):
    """
    可选：若需要 ratio 量化（2D/3D 都可用 size[..., :2]）
    size: [B,2] or [B,3]（只用前两个分量）
    """
    default_ratios = default_aspect_ratios.to(device)
    h = size[..., 0:1]
    w = size[..., 1:2]
    ratio = h / (w + 1e-6)
    distance = torch.abs(ratio - default_ratios.view(1, -1))  # [B, n]
    idx = torch.argmin(distance, dim=-1)
    ratio_quant = default_ratios[idx]
    return ratio_quant


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    inputs: [N] logits
    targets: [N] in {0,1}
    """
    targets = targets.float()
    device = targets.device

    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, pt, 1 - pt)

    alpha_t = torch.where(targets == 1, 1 - alpha, alpha).to(device)
    loss = alpha_t * (1 - pt) ** gamma * bce
    return loss.mean()


def BCELogitsLoss_with_HNM(pred_prob, gt_prob, positive, gt_before_query, weight):
    """
    与 2D 版一致，保留 HNM
    """
    b, t, N = pred_prob.shape
    gt_prob = gt_prob.unsqueeze(-1).repeat(1, 1, N)   # [b,t,N]

    pred_prob_v = rearrange(pred_prob, 'b t N -> (b t N)')
    gt_prob_v   = rearrange(gt_prob,   'b t N -> (b t N)')
    bce = F.binary_cross_entropy_with_logits(pred_prob_v, gt_prob_v, reduction='none')

    pred_prob = rearrange(pred_prob_v, '(b t N) -> b t N', b=b, t=t)
    gt_prob   = rearrange(gt_prob_v,   '(b t N) -> b t N', b=b, t=t)
    bce       = rearrange(bce,         '(b t N) -> b t N', b=b, t=t)
    positive  = rearrange(positive,    '(b t N) -> b t N', b=b, t=t)

    loss = HardNegMining(pred_prob, gt_prob, positive, bce, gt_before_query, weight)
    return loss.mean()


def HardNegMining(pred_prob, gt_prob, positive, BCE_loss, gt_before_query, weight, ratio_neg_pos=3., ratio_hard=0.05):
    """
    与 2D 版一致
    """
    b, t, N = pred_prob.shape
    b_real = int(b ** 0.5)
    w_pos, w_neg = weight

    mined = []
    for i in range(b_real):
        query_idx = [(i + j * b_real) for j in range(b_real)]
        cur_valid = gt_before_query[query_idx].bool()

        cur_pos = positive[query_idx][cur_valid]
        cur_loss = BCE_loss[query_idx][cur_valid]
        M = cur_loss.shape[0]

        num_pos = int(torch.sum(cur_pos).item())
        num_neg = int(ratio_neg_pos * num_pos) if num_pos > 0 else int(ratio_hard * M)

        pos_losses = cur_loss[cur_pos.bool()]
        neg_losses = cur_loss[~cur_pos.bool()]
        num_neg = min(num_neg, neg_losses.shape[0])
        hard_negs, _ = torch.topk(neg_losses, num_neg)

        mined.append(pos_losses * w_pos)
        mined.append(hard_negs * w_neg)

    mined = torch.cat(mined, dim=0) if len(mined) > 0 else torch.zeros(1, device=pred_prob.device)
    return mined
