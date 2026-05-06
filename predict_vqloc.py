#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict & dump eval cache (3D)

- 使用你训练时相同的数据流：dataset.VQLOC + dataset_utils.process_data
- 不改动你的数据集代码；仅在本脚本里 monkey-patch 父类 QueryVideoDataset 的 __init__ 与 _load_metadata
- batch_size=1，逐样本推理
"""

import os
import json
import pprint
import random
import argparse
import numpy as np
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader

# ==== 项目内 ====
from config.config import config, update_config
from utils import exp_utils
from dataset.VQLOC import VQLOC
from dataset import dataset_utils
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher


# -------------------
# 解析参数
# -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Predict & dump eval cache (3D)")
    parser.add_argument("--cfg", required=True, type=str, help="config yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", default=None, type=str, help="覆盖数据根目录")
    parser.add_argument("--ckpt", default="", type=str, help="模型权重路径")
    parser.add_argument("--gpus", default="", type=str, help="unused (兼容)")
    parser.add_argument("--workers_per_gpu", default=4, type=int)
    parser.add_argument("--limit", default=0, type=int, help="只跑前 N 个样本（0=全部）")
    # 兼容你之前命令里的别名参数，避免报错
    parser.add_argument("--bs", default=1, type=int, help="ignored: 预测固定逐样本")
    parser.add_argument("--nw", default=0, type=int, help="ignored: 请用 --workers_per_gpu")
    parser.add_argument("--vis_every", default=0, type=int, help="ignored: 预测阶段默认不出图")
    parser.add_argument("--save_preds", default="./output/vqloc_pred.json", type=str)
    args = parser.parse_args()
    update_config(args.cfg)
    return args


# -------------------
# 只在本脚本中：屏蔽父类 metadata 加载 + 父类 __init__
# （不改任何数据集文件）
# -------------------
def _monkey_patch_query_video_dataset():
    try:
        from dataset.base_dataset import QueryVideoDataset
    except Exception:
        return

    # 1) 覆盖 _load_metadata，避免去找 VQ2D 的 json
    def _noop_load(self):
        self.metadata = None
        return

    # 2) 覆盖 __init__，只做最小化属性设置，不碰 annotations
    def _patched_init(self, dataset_name, query_params, clip_params, data_dir, split="train"):
        self.dataset_name = dataset_name
        self.query_params = dict(query_params) if query_params is not None else {}
        self.clip_params = dict(clip_params) if clip_params is not None else {}
        self.data_dir = data_dir
        self.split = split
        # 训练里你在 VQLOC 里用到的：self.padding_value
        pv = self.clip_params.get("padding_value", "mean")
        # 以 0.5 近似 "mean"（ToTensor 后是 0~1），"zero" 则 0.0
        self.padding_value = 0.0 if str(pv).lower() == "zero" else 0.5
        # 兼容一些占位字段
        self.tracklets = False
        self.metadata = None
        # 不做任何关于 self.annotations 的操作

    QueryVideoDataset._load_metadata = _noop_load
    QueryVideoDataset.__init__ = _patched_init


# -------------------
# 构建数据集（与训练一致）
# -------------------
def _cfg_dataset_root(cfg) -> str:
    # 优先命令行 data_dir，其次 cfg.dataset.data_dir/root
    root = getattr(cfg.dataset, "data_dir", None) or getattr(cfg.dataset, "root", None)
    return root

def build_dataset(cfg, split: str, data_dir: str):
    if split == "train":
        clip_num_frames = int(getattr(cfg.dataset, "clip_num_frames", 30))
    else:
        clip_num_frames = int(getattr(cfg.dataset, "clip_num_frames_val",
                                      getattr(cfg.dataset, "clip_num_frames", 30)))
    clip_params = {
        "fine_size": int(getattr(cfg.dataset, "clip_size_fine",
                                 getattr(cfg.dataset, "clip_size_coarse", 448))),
        "frame_npts": int(getattr(cfg.dataset, "frame_npts", 1024)),
        "frame_interval": int(getattr(cfg.dataset, "frame_interval", 1)),
        "clip_num_frames": clip_num_frames,
        "padding_value": getattr(cfg.dataset, "padding_value", "mean"),  # 父类需要
    }
    query_params = {
        "query_square": bool(getattr(cfg.dataset, "query_square", True)),
        "query_padding": bool(getattr(cfg.dataset, "query_padding", False)),
        "query_size": int(getattr(cfg.dataset, "query_size",
                                  getattr(cfg.dataset, "clip_size_fine",
                                          getattr(cfg.dataset, "clip_size_coarse", 448)))),
    }
    ds = VQLOC(
        dataset_name=getattr(cfg.dataset, "name", "VQLOC"),
        query_params=query_params,
        clip_params=clip_params,
        data_dir=data_dir,
        split=split,
    )
    return ds


# -------------------
# 构建模型并加载权重
# -------------------
def build_model(cfg, device, ckpt_path: str):
    model = ClipMatcher(cfg).to(device)
    print('Model params:', sum(p.numel() for p in model.parameters()))
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Resume] missing={list(missing.keys()) if hasattr(missing,'keys') else missing}, "
              f"unexpected={list(unexpected.keys()) if hasattr(unexpected,'keys') else unexpected}")
        del ckpt, state
    else:
        print(f"[Warn] 未提供或找不到 ckpt: {ckpt_path}")
    model.eval()
    return model


# -------------------
# 取 top1（和训练的 val_performance 思路一致）
# -------------------
@torch.no_grad()
def pick_top1(preds: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    preds:
      prob: [B,T,N]
      bbox: [B,T,N,C]  (C=9 for 3D)
    返回每条样本的 per-frame top1 bbox(9) 与 prob(T)
    """
    prob = preds["prob"]                # [B,T,N]
    bbox = preds["bbox"]                # [B,T,N,C]
    B, T, N = prob.shape
    C = bbox.shape[-1]
    prob_bt = prob.reshape(B*T, N)
    bbox_bt = bbox.reshape(B*T, N, C)
    scores, idx = torch.max(prob_bt, dim=-1)           # [(B*T)]
    gather_idx = idx.view(-1, 1, 1).repeat(1, 1, C)
    top_bbox_bt = torch.gather(bbox_bt, 1, gather_idx).squeeze(1)  # [(B*T), C]
    top_bbox = top_bbox_bt.view(B, T, C)              # [B,T,C]
    top_prob = torch.sigmoid(scores).view(B, T)       # [B,T]
    return {"bbox": top_bbox, "prob": top_prob}


# -------------------
# 主流程
# -------------------
def main():
    args = parse_args()
    print(args)

    # 先打补丁（避免父类提前碰 annotations / 去找外部 JSON）
    _monkey_patch_query_video_dataset()

    # 日志
    logger, output_dir, _ = exp_utils.create_logger(config, args.cfg, phase='eval')
    config.output_dir = output_dir
    logger.info(pprint.pformat(config))

    # 随机种子
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # 数据根目录
    data_root = args.data_dir or _cfg_dataset_root(config)
    if not data_root or not os.path.isdir(data_root):
        raise ValueError(f"无效的数据目录：{data_root}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集 & DataLoader（逐样本）
    ds = build_dataset(config, args.split, data_root)
    if args.limit and args.limit > 0:
        ds.annotations = ds.annotations[:args.limit]  # 只切片你自己准备好的 annotations

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers_per_gpu,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: batch[0],  # 不要把 dict 堆叠坏
        drop_last=False,
    )

    # 模型
    model = build_model(config, device, args.ckpt)

    # 预测累积
    os.makedirs(os.path.dirname(args.save_preds), exist_ok=True)
    all_preds: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i, sample in enumerate(loader):
            # 与训练相同的预处理 & H2D
            sample = exp_utils.dict_to_cuda(sample, device=device)
            sample = dataset_utils.process_data(config, sample, split="val", device=device)

            clips = sample["clip"]                # [T,3,H,W]
            if config.train.use_query_roi and ("query_frame" in sample):
                preds = model(
                    clips,
                    sample["query_frame"],
                    query_frame_bbox=sample["query_frame_bbox"],
                    training=False,
                    fix_backbone=config.model.fix_backbone,
                )
            else:
                preds = model(
                    clips, sample["query"], training=False,
                    fix_backbone=config.model.fix_backbone
                )

            top = pick_top1(preds)  # {"bbox":[1,T,C], "prob":[1,T]}
            top_bbox = top["bbox"].squeeze(0).cpu().tolist()  # [T,C]
            top_prob = top["prob"].squeeze(0).cpu().tolist()  # [T]

            # 基本索引信息，方便与你的评测对齐用
            meta = {}
            try:
                ann = ds.annotations[i]
                meta = {
                    "batch_id": int(ann.get("batch_id", -1)),
                    "scene_id": int(ann.get("scene_id", -1)),
                    "n_frames": len(ann.get("search", [])),
                }
            except Exception:
                meta = {"index": i}

            all_preds.append({
                **meta,
                "top_bbox_3d": top_bbox,   # 每帧 9 维: [cx,cy,cz,l,w,h,roll,pitch,yaw]
                "top_prob": top_prob,      # 每帧得分(0~1)
            })

            if (i + 1) % 50 == 0:
                logger.info(f"[{i+1}/{len(ds)}] done")

    with open(args.save_preds, "w") as f:
        json.dump(all_preds, f)
    print(f"[Done] saved predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
