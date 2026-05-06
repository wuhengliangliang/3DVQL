import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchvision import transforms
from einops import rearrange
# 3D/2D 统一的 assign_labels（你上面已经改为同时支持 4/9 维）
from utils.anchor_utils import assign_labels
from dataset import dataset_utils

# 如需 2D ratio（可选，用于某些分支），保留默认表
default_aspect_ratios = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])

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
    center_point      = preds['center_point']        # [1,1,N,9]
    if 'prob_refine' in preds.keys():
        pred_prob_refine = preds['prob_refine']   # [b,t]

    b, t, N = pred_prob.shape
    device = pred_prob.device

    # GT 取 3D 接口（若缺失则从 2D 回退，仅在数据集仍给 2D 时使用）
    if 'clip_pcd_bbox' in gts:
        gt_bbox = gts['clip_pcd_bbox']           # [b,t,9]
        if 'center' not in gts.keys():
            gts['center'] = gt_bbox[..., :3]
        if 'size' not in gts.keys():
            gts['size']   = gt_bbox[..., 3:6]
        if 'rot' not in gts.keys():
            gts['rot']    = gt_bbox[..., 6:9]
    else:
        # 回退：若只提供 2D（不推荐，保持兼容）
        gt_bbox_2d = gts['clip_bbox']             # [b,t,4]
        # 构造一个假的 3D 以免下游崩（z/lwh/角度置 0）
        zeros_3 = torch.zeros_like(gt_bbox_2d[..., :1]).repeat(1,1,3)
        zeros_r = torch.zeros_like(gt_bbox_2d[..., :1]).repeat(1,1,3)
        # 用 2D 中心和 hw 填入 x,y；z=0; l,w,h=0; roll/pitch/yaw=0（仅兼容用途）
        cxcy = (gt_bbox_2d[..., :2] + gt_bbox_2d[..., 2:]) / 2.0
        # 拼成 [x,y,z,l,w,h,roll,pitch,yaw]
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

    if gt_prob.bool().any():
        assign_label = assign_labels(
            center_points=center_point.repeat(b, t, 1, 1),           # [b,t,N,9]
            gt_center=gt_center,                              # [b,t,9]
            distance_threshold=config.model.positive_threshold,
            topk=config.model.positive_topk
        )                                        # [b,t,N]
        positive = torch.logical_and(
            gt_prob.unsqueeze(-1).repeat(1,1,N).bool(),
            assign_label.bool()
        )                                        # [b,t,N]
        positive = rearrange(positive, 'b t N -> (b t N)')
    else:
        positive = torch.zeros(b, t, N, device=device).reshape(-1).bool()

    if torch.sum(positive.float()).item() == 0:
        positive[:1] = True
    loss_mask = positive.float().unsqueeze(1)    # [b*t*N,1]

    # 回归损失
    if torch.sum(positive.float()).item() > 0:
        # center
        pred_center = rearrange(pred_center, 'b t N c -> (b t N) c')
        gt_center_rep = rearrange(gt_center.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_center = F.l1_loss(pred_center[positive], gt_center_rep[positive])

        # size (half-size)
        pred_size = rearrange(pred_size, 'b t N c -> (b t N) c')
        gt_size_rep = rearrange(gt_size.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_size = F.l1_loss(pred_size[positive], gt_size_rep[positive])

        # rot
        pred_rot = rearrange(pred_rot, 'b t N c -> (b t N) c')
        gt_rot_rep = rearrange(gt_rot.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_rot = F.l1_loss(pred_rot[positive], gt_rot_rep[positive])

        # bbox distance loss
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t N) c')
        # pred_center = rearrange(pred_center, 'b t N c -> (b t N) c')
        gt_center_replicate = rearrange(gt_center.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        loss_distance = DistanceLoss(pred_center, gt_center_replicate, mask=loss_mask.bool().squeeze()) # [b*t*N]

        # 展平回 [b*t*N,c] 方便后面取 top
        # pred_bbox = rearrange(pred_bbox_btNc, '(b t) N c -> (b t N) c', b=b, t=t)
    else:
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t N) c')
        loss_center = torch.tensor(0.0, device=device)
        loss_size   = torch.tensor(0.0, device=device)
        loss_rot    = torch.tensor(0.0, device=device)
        loss_distance   = torch.tensor(0.0, device=device)

    # 发生概率（anchor hit）- focal loss on anchors for valid frames
    pred_prob_vec = rearrange(pred_prob, 'b t N -> (b t N)')
    valid_mask = rearrange(gt_before_query.unsqueeze(2).repeat(1,1,N), 'b t N -> (b t N)')
    loss_prob = focal_loss(
        pred_prob_vec[valid_mask].float(),
        positive[valid_mask].float()
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
        'loss_bbox_distance':   loss_distance,
        'loss_prob':        loss_prob,
        # weights
        'weight_bbox_center': config.loss.weight_bbox_center,
        'weight_bbox_size':   config.loss.weight_bbox_size,
        'weight_bbox_rot':    config.loss.weight_bbox_rot,   # 修正键名
        'weight_bbox_distance':   config.loss.weight_bbox_distance,
        'weight_prob':        config.loss.weight_prob,
        # information
        # 'iou':  iou.detach(),
        # 'giou': giou.detach()
    }
    if 'prob_refine' in preds.keys():
        loss.update({
            'loss_prob_refine':  loss_prob_refine,
            'weight_prob_refine': 1.0
        })

    # 取每帧 top1 预测（按 prob）
    pred_prob_btN = rearrange(pred_prob_vec, '(B N) -> B N', N=N)       # [b*t,N]
    pred_bbox_btNc = rearrange(pred_bbox, '(B N) c -> B N c', N=N)      # [b*t,N,9]
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

    # bbox losses
    loss_center = F.l1_loss(pred_center[gt_prob.bool()], gt_center[gt_prob.bool()])
    loss_size   = F.l1_loss(pred_size[gt_prob.bool()],   gt_size[gt_prob.bool()])
    loss_rot    = F.l1_loss(pred_rot[gt_prob.bool()],    gt_rot[gt_prob.bool()])

    loss_distance = DistanceLoss(pred_center, gt_center, mask=gt_prob.bool())

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
        'loss_bbox_distance':   loss_distance,
        'loss_prob':        loss_prob,
        # weights
        'weight_bbox_center': config.loss.weight_bbox_center,
        'weight_bbox_size':   config.loss.weight_bbox_size,
        'weight_bbox_rot':    config.loss.weight_bbox_rot,   # 修正键名
        'weight_bbox_distance':   config.loss.weight_bbox_distance,
        'weight_prob':        config.loss.weight_prob,
        # info
    }
    return loss

def DistanceLoss(center_points, gt_center, mask=None):
    '''
    center_points: in shape [N,3]
    gt_center: in shape [N,3]
    mask: ground truth of valid instance, in shape [N]
    return:
        distance: in shape [N,]
    '''
    device = center_points.device
    N = center_points.shape[0]
    
    distance = torch.norm(center_points - gt_center, dim=-1)  # shape: [N,]
    if torch.is_tensor(mask):
        distance = torch.mean(distance[mask])
    else:
        distance = torch.mean(distance)
    return distance

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
