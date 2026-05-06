# func/train_anchor.py
import os
import time
import logging
import random
import itertools
import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

from utils import exp_utils, train_utils, loss_utils, vis_utils
from dataset import dataset_utils
from utils.loss_utils import DistanceLoss
from utils.anchor_utils import calculate_iou, calculate_distance
logger = logging.getLogger(__name__)


def _ensure_sample_bool_masks(sample: dict):
    """
    把样本里常见的 mask 统一转成 torch.bool，避免 Transformer 的 dtype 警告与性能下降。
    不存在的 key 会被忽略，存在且不是 Tensor 的也会忽略。
    """
    mask_keys = [
        "attn_mask", "src_key_padding_mask", "key_padding_mask",
        "clip_pad_mask", "query_pad_mask", "clip_mask", "padding_mask",
        "video_pad_mask", "frame_pad_mask"
    ]
    for k in mask_keys:
        v = sample.get(k, None)
        if isinstance(v, torch.Tensor) and v.dtype is not torch.bool:
            sample[k] = v.to(torch.bool)
    return sample


def train_epoch(
    config,
    loader,
    model,
    optimizer,
    schedular,
    scaler,
    epoch,
    output_dir,
    device,
    rank,
    wandb_run=None,
    ddp=True,
):
    
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()

    train_utils.set_model_train(config, model, ddp)

    is_ddp = ddp and isinstance(model, torch.nn.parallel.DistributedDataParallel)
    ddp_module = model if not is_ddp else model

    preprocess_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    # —— 本 epoch 可视化上限（防止 I/O 把训练卡死）——
    # max_train_vis = int(getattr(config.train, "max_train_vis", 2))
    # vis_done = 0

    for batch_idx, sample in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch
        iter_start = time.time()

        # ========= H2D + 预处理（放独立 stream）=========
        t0 = time.time()
        if device.type == "cuda" and preprocess_stream is not None:
            with torch.cuda.stream(preprocess_stream):
                sample = exp_utils.dict_to_cuda(sample)
                sample = dataset_utils.process_data(
                    config, sample, iter=iter_num, split="train", device=device
                )
                sample = _ensure_sample_bool_masks(sample)
            torch.cuda.current_stream(device).wait_stream(preprocess_stream)
        else:
            sample = exp_utils.dict_to_cuda(sample)
            sample = dataset_utils.process_data(
                config, sample, iter=iter_num, split="train", device=device
            )
            sample = _ensure_sample_bool_masks(sample)
        time_meters.add_loss_value("Data time", time.time() - t0)
        # ==============================================

        clips_img, query_img = sample['clip'], sample['query']
        clips_pcd, query_pcd = sample['clip_pcd'], sample['query_frame_pcd']

        use_no_sync = is_ddp and ((batch_idx + 1) % config.train.accumulation_step != 0)
        sync_ctx = ddp_module.no_sync if use_no_sync else contextlib.nullcontext

        with sync_ctx():
            # 前向
            t1 = time.time()
            preds = model(clips_img, query_img, sample['query_frame_bbox'], 
                                    clips_pcd, query_pcd, sample['query_frame_pcd_bbox'], 
                                    training=True, fix_backbone=config.model.fix_backbone)
            time_meters.add_loss_value("Prediction time", time.time() - t1)

            # 损失
            losses, preds_top, sample = loss_utils.get_losses_with_anchor(config, preds, sample)
            total_loss = 0.0
            for k, v in losses.items():
                if "loss" in k:
                    total_loss += losses[k.replace("loss_", "weight_")] * v
                    loss_meters.add_loss_value(k, float(v.detach().item()))
            total_loss = total_loss / config.train.accumulation_step

            # 反向
            total_loss.backward()

        # 累积边界再 step
        if (batch_idx + 1) % config.train.accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.train.grad_max, norm_type=2.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedular.step()

        # 打印
        if iter_num % config.print_freq == 0:
            batch_time = time.time() - iter_start
            time_meters.add_loss_value("Batch time", batch_time)
            msg = (
                f"Epoch {epoch}, Iter {iter_num}, rank {rank}, "
                f"Time: data {time_meters.average_meters['Data time'].val:.3f}s, "
                f"pred {time_meters.average_meters['Prediction time'].val:.3f}s, "
                f"all {batch_time:.3f}s ({time_meters.average_meters['Batch time'].avg:.3f}s), "
                f"Loss: "
            )
            for k, v in loss_meters.average_meters.items():
                msg += f"{k}: {v.val:.6f} ({v.avg:.6f}), "
            logger.info(msg[:-2])

        # —— 可视化（仅 rank0，受 freq + 上限控制）——
        do_vis = (
            (rank == 0)
            and (getattr(config, "vis_freq", 0) > 0)
            # and (vis_done < max_train_vis)
            and (iter_num % getattr(config, "vis_freq", 0) == 0)
        )
        if do_vis:
            try:
                vis_utils.vis_pred_clip(
                    sample=sample, preds=[preds_top], iter_num=iter_num,
                    output_dir=output_dir, subfolder="train"
                )
            except Exception as e:
                print(f"vis_pred_clip failed: {e}")
            try:
                vis_utils.vis_pred_scores(
                    sample=sample, preds=[preds_top], iter_num=iter_num,
                    output_dir=output_dir, subfolder="train"
                )
            except Exception as e:
                print(f"vis_pred_scores failed: {e}")
            # vis_done += 1

        # wandb（仅 rank0）
        if (rank == 0) and (wandb_run is not None):
            wandb_log = {
                "Train/loss": float(total_loss.item()),
                "Train/lr": optimizer.param_groups[0]["lr"],
            }
            for k, v in losses.items():
                if "loss" in k:
                    wandb_log[f"Train/{k}"] = float(v.item())
            wandb_run.log(wandb_log)

        if batch_idx < 3:
            torch.cuda.empty_cache()



@torch.no_grad()
def validate(
    config,
    loader,
    model,
    epoch,
    output_dir,
    device,
    rank,
    wandb_run=None,
    ddp=True,
):
    train_utils.set_model_eval(config, model, ddp)

    # —— 只 rank0 画图；频率+上限控制；可限制验证批次数 —— 
    do_vis = (rank == 0) and (getattr(config, "eval_vis_freq", 0) > 0)
    eval_vis_freq = int(getattr(config, "eval_vis_freq", 0))
    eval_max_vis = int(getattr(config, "eval_max_vis", 2))
    vis_done = 0

    is_ddp = ddp and isinstance(model, torch.nn.parallel.DistributedDataParallel)
    ddp_module = model if not is_ddp else model
    preprocess_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    max_val_batches = int(getattr(config.test, "max_val_batches", -1))

    # 聚合指标：对每个 key 统计 sum 和 count，最后 all_reduce
    sum_dict = {}
    cnt_dict = {}

    for batch_idx, sample in enumerate(loader):
        if max_val_batches > 0 and batch_idx >= max_val_batches:
            break

        # ========= H2D + 预处理（放独立 stream）=========
        if device.type == "cuda" and preprocess_stream is not None:
            with torch.cuda.stream(preprocess_stream):
                sample = exp_utils.dict_to_cuda(sample)
                sample = dataset_utils.process_data(
                    config, sample, split="val", device=device
                )
                sample = _ensure_sample_bool_masks(sample)
            torch.cuda.current_stream(device).wait_stream(preprocess_stream)
        else:
            sample = exp_utils.dict_to_cuda(sample)
            sample = dataset_utils.process_data(
                config, sample, split="val", device=device
            )
            sample = _ensure_sample_bool_masks(sample)
        # ==============================================
        # # 预处理 + 统一布尔 mask
        # sample = exp_utils.dict_to_cuda(sample)
        # sample = dataset_utils.process_data(config, sample, split="val", device=device)
        # sample = _ensure_sample_bool_masks(sample)

        clips_img, query_img = sample['clip'], sample['query']
        clips_pcd, query_pcd = sample['clip_pcd'], sample['query_frame_pcd']

        use_no_sync = is_ddp and ((batch_idx + 1) % config.train.accumulation_step != 0)
        sync_ctx = ddp_module.no_sync if use_no_sync else contextlib.nullcontext
        
        with sync_ctx():
            # 前向
            preds = model(clips_img, query_img, sample['query_frame_bbox'], 
                            clips_pcd, query_pcd, sample['query_frame_pcd_bbox'], 
                            training=False, fix_backbone=config.model.fix_backbone)
            results, preds_top = val_performance(config, preds, sample)

        

        # 累加 sum / count
        for k, v in results.items():
            sum_dict[k] = sum_dict.get(k, 0.0) + float(v)
            cnt_dict[k] = cnt_dict.get(k, 0) + 1

        # —— 验证可视化（仅 rank0 + 频率 + 上限）——
        if do_vis and (vis_done < eval_max_vis) and (batch_idx % eval_vis_freq == 0):
            try:
                vis_utils.vis_pred_clip(
                    sample=sample, preds=[preds_top], iter_num=batch_idx,
                    output_dir=output_dir, subfolder="val"
                )
            except Exception as e:
                print(f"vis_pred_clip failed (val): {e}")
            try:
                vis_utils.vis_pred_scores(
                    sample=sample, preds=[preds_top], iter_num=batch_idx,
                    output_dir=output_dir, subfolder="val"
                )
            except Exception as e:
                print(f"vis_pred_scores failed (val): {e}")
            vis_done += 1

    # ========== DDP 聚合 ==========
    keys = sorted(sum_dict.keys())
    if len(keys) == 0:
        # 没有任何度量，返回 0
        iou, prob_acc = 0.0, 0.0
        return iou, prob_acc

    local_sum = torch.tensor([sum_dict[k] for k in keys], device=device, dtype=torch.float32)
    local_cnt = torch.tensor([float(cnt_dict[k]) for k in keys], device=device, dtype=torch.float32)

    if ddp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)

    avg = (local_sum / torch.clamp(local_cnt, min=1.0)).tolist()
    avg_dict = {k: a for k, a in zip(keys, avg)}

    # 记录 wandb（仅 rank0）
    if (rank == 0) and (wandb_run is not None):
        wandb_run.log({f"Valid/{k}": v for k, v in avg_dict.items()})

    # 返回主要两个指标
    iou = float(avg_dict.get("iou", 0.0))
    prob_acc = float(avg_dict.get("prob_accuracy", 0.0))
    return iou, prob_acc



def _mean_or_zero(x):
    return float(x.mean().item()) if x.numel() > 0 else 0.0


def val_performance(config, preds, gts, prob_theta=0.5):
    """
    同时支持：
    - 2D：preds['bbox'] 最末维 4；可选 'center'、'hw'
    - 3D：preds['bbox'] 最末维 9；可选 'center'(3)、'size'(3)、'rot'(3)
    """
    assert "prob" in preds and "bbox" in preds, "preds 必须包含 'prob' 和 'bbox'"
    pred_prob = preds["prob"]           # [b,t,N]
    pred_bbox = preds["bbox"]           # [b,t,N,C], C=4 或 9

    b, t, N = pred_prob.shape
    C = pred_bbox.shape[-1]

    # 展开并取 top-1 proposal
    pp = rearrange(pred_prob, "b t N -> (b t) N")
    pb = rearrange(pred_bbox, "b t N c -> (b t) N c")

    top_scores, top_idx = torch.max(pp, dim=-1)  # [(b*t)]
    gather_idx = top_idx.view(-1, 1, 1).repeat(1, 1, C)
    top_bbox = torch.gather(pb, dim=1, index=gather_idx).squeeze(1)  # [(b*t),C]

    # GT
    gt_bbox_3d = rearrange(gts["clip_pcd_bbox"], "b t c -> (b t) c")
    gt_bbox = gt_bbox_3d
    gt_center3 = gt_bbox_3d[:, 0:3]
    gt_size3 = gt_bbox_3d[:, 3:6]
    gt_rot3 = gt_bbox_3d[:, 6:9]

    gt_prob = gts["clip_with_bbox"].reshape(-1)
    gt_before_query = gts["before_query"].reshape(-1)
    mask = gt_prob.bool()

    if all(k in preds for k in ["center", "size", "rot"]):
        pc = rearrange(preds["center"], "b t N c -> (b t) N c")
        ps = rearrange(preds["size"], "b t N c -> (b t) N c")
        pr = rearrange(preds["rot"], "b t N c -> (b t) N c")
        pc = torch.gather(pc, 1, top_idx.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
        ps = torch.gather(ps, 1, top_idx.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
        pr = torch.gather(pr, 1, top_idx.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
    else:
        pc = top_bbox[:, 0:3]
        ps = top_bbox[:, 3:6]
        pr = top_bbox[:, 6:9]

    loss_center = F.l1_loss(pc[mask], gt_center3[mask]) if mask.any() else torch.tensor(0., device=gt_bbox.device)
    loss_size = F.l1_loss(ps[mask], gt_size3[mask]) if mask.any() else torch.tensor(0., device=gt_bbox.device)
    loss_rot = F.l1_loss(pr[mask], gt_rot3[mask]) if mask.any() else torch.tensor(0., device=gt_bbox.device)

    loss_distance = DistanceLoss(pc, gt_center3, mask=mask)
    iou_all = calculate_iou(top_bbox, gt_bbox)  # [(b*t)]
    iou = _mean_or_zero(iou_all[mask]) if mask.any() else 0.0
    iou_25 = float(((iou_all[mask] > 0.25).float().mean().item())) if mask.any() else 0.0

    loss_dict = {
        "loss_bbox_center": float(loss_center.item()),
        "loss_bbox_size": float(loss_size.item()),
        "loss_bbox_rot": float(loss_rot.item()),
        "loss_bbox_distance": float(loss_distance.item()),
        "iou": iou,
        "iou_25": iou_25,
    }

    # 概率指标（兼容 prob_refine）
    if "prob_refine" in preds:
        prob_for_metric = preds["prob_refine"].reshape(-1)
    else:
        prob_for_metric = top_scores

    prob_accuracy = ((torch.sigmoid(prob_for_metric) > prob_theta) == gt_prob.bool()).float().mean()
    prob_accuracy_2 = ((torch.sigmoid(prob_for_metric) > 0.6) == gt_prob.bool()).float().mean()
    prob_accuracy_3 = ((torch.sigmoid(prob_for_metric) > 0.7) == gt_prob.bool()).float().mean()
    prob_accuracy_4 = ((torch.sigmoid(prob_for_metric) > 0.65) == gt_prob.bool()).float().mean()

    loss_dict.update({
        "loss_prob": 0.0,
        "prob_accuracy": float(prob_accuracy.item()),
        "prob_accuracy_0.6": float(prob_accuracy_2.item()),
        "prob_accuracy_0.7": float(prob_accuracy_3.item()),
        "prob_accuracy_0.65": float(prob_accuracy_4.item()),
    })

    # 可视化需要
    top_prob_bt = torch.sigmoid(top_scores).view(b, t)
    top_bbox_btC = top_bbox.view(b, t, C)
    pred_top = {"bbox": top_bbox_btC, "prob": top_prob_bt}

    return loss_dict, pred_top
