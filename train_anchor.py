# -*- coding: utf-8 -*-
import os
# if ("LOCAL_RANK" not in os.environ and "RANK" not in os.environ
#         and "CUDA_VISIBLE_DEVICES" not in os.environ):
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import pprint
import random
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from config.config import config, update_config
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm_ddp import ClipMatcher

from utils import exp_utils, train_utils, dist_utils
from dataset import dataset_utils
from func.train_anchor import train_epoch, validate

import transformers
import wandb

# ========================= 全局环境 & 日志降噪 =========================
# 1) 避免 cuDNN 引擎选择导致的 FIND/GET 异常
torch.backends.cudnn.enabled = False

# 2) 允许 TF32（A100/Hopper 上有加速）
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# 3) 屏蔽 transformer 的 mask dtype 警告
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance.",
    module="torch.nn.modules.transformer",
)

# 4) 降低 NCCL 日志量
os.environ.setdefault("NCCL_DEBUG", "WARN")

# ========================= 工具函数 =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train anchor model")
    parser.add_argument(
        "--cfg",
        required=False,
        type=str,
        default="./config/train.yaml",
        help="path to yaml config",
    )
    parser.add_argument("--eval", dest="eval", action="store_true", help="evaluate model only")
    parser.add_argument("--local_rank", default=-1, type=int, help="local rank for torchrun")
    args, _ = parser.parse_known_args()
    update_config(args.cfg)
    return args


def build_dataloaders(ddp):
    train_data = dataset_utils.get_dataset(config, split="train")
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.train.batch_size,
        shuffle=shuffle_flag,
        num_workers=int(config.workers),
        pin_memory=True,
        persistent_workers=(int(config.workers) > 0),
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=2 if int(config.workers) > 0 else None,
    )

    val_data = dataset_utils.get_dataset(config, split="val")
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=int(config.workers),
        pin_memory=True,
        persistent_workers=(int(config.workers) > 0),
        drop_last=False,
        prefetch_factor=2 if int(config.workers) > 0 else None,
    )
    return train_loader, val_loader, train_sampler


def _replace_deconv_in_module(module: nn.Module):
    """
    遍历模型，把 ConvTranspose2d 统一替换为 Upsample(scale=2)+Conv2d。
    这样可以完全避开某些环境下 conv_transpose2d 触发的 FIND 引擎报错。
    替换过程在 .to(device) 之前进行；或者替换完成后再 .to(device) 一次。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            # 用最近邻上采样 + 3x3 Conv2d 近似替代反卷积上采样
            stride = child.stride if isinstance(child.stride, tuple) else (child.stride, child.stride)
            scale = stride[0]  # 常见是 2
            new_block = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="nearest"),
                nn.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=(child.bias is not None),
                ),
            )
            setattr(module, name, new_block)
        else:
            _replace_deconv_in_module(child)


def try_resume_safely(model, optimizer, schedular, scaler, output_dir, rank=0):
    """
    更健壮的恢复策略：
      - 模型：strict=False（允许 key 有出入，比如替换/新增层）
      - 优化器/调度器：若加载失败，打印原因并重建；epoch 仍按 ckpt 继续
    """
    ckpt_path = os.path.join(output_dir, "cpt_last.pth.tar")
    if not os.path.isfile(ckpt_path):
        if rank == 0:
            print(f"[Resume] => no checkpoint found at '{ckpt_path}'")
        return model, optimizer, schedular, scaler, None, 0.0, 0.0

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 1) 模型
    missing, unexpected = model.load_state_dict(checkpoint.get("state_dict", {}), strict=False)
    if rank == 0:
        print(f"[Resume] Loaded: {ckpt_path}")
        print(f"[Resume] missing={list(missing)}, unexpected={list(unexpected)}")

    # 2) 训练进度 & 记录
    best_iou = checkpoint.get("best_iou", 0.0)
    best_prob = checkpoint.get("best_prob", 0.0)
    ep_resume = checkpoint.get("epoch", 0)

    # 3) 优化器
    opt_ok = False
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
        opt_ok = True
    except Exception as e:
        if rank == 0:
            print(f"[Resume] optimizer state not loaded, rebuild optimizer. reason: {e}")

    # 4) 调度器
    try:
        schedular.load_state_dict(checkpoint["schedular"])
    except Exception as e:
        if rank == 0:
            print(f"[Resume] scheduler state not loaded, continue with current. reason: {e}")

    # 5) AMP scaler（如果你在 train_epoch 里没用 AMP，加载失败也无所谓）
    try:
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    except Exception as e:
        if rank == 0:
            print(f"[Resume] scaler state not loaded, continue fresh. reason: {e}")

    if rank == 0:
        if opt_ok:
            print(f"[Resume] done. restart from epoch {ep_resume}")
        else:
            print(f"[Resume] model weights restored. optimizer reset. restart from epoch {ep_resume}")

    return model, optimizer, schedular, scaler, ep_resume, best_iou, best_prob


def main():
    # 单机默认只绑一张卡，避免误占
    if ("LOCAL_RANK" not in os.environ and "RANK" not in os.environ
            and "CUDA_VISIBLE_DEVICES" not in os.environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parse_args()

    # logger
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase="train")
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # 随机种子
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # DDP 初始化
    using_torchrun = ("LOCAL_RANK" in os.environ) or ("RANK" in os.environ)
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if num_gpus > 0 else "cpu")

    if using_torchrun:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist_utils.dist_init(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    # wandb（只在主进程启用）
    if rank == 0:
        wandb_run = wandb.init(project=config.exp_group, group=config.exp_name, mode="disabled")
        wandb.config.update({
            "exp_name": config.exp_name,
            "batch_size": config.train.batch_size,
            "total_iteration": config.train.total_iteration,
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "loss_weight_bbox_giou": config.loss.weight_bbox_giou,
            "loss_prob_bce_weight": config.loss.prob_bce_weight,
            "model_num_transformer": config.model.num_transformer,
            "model_resolution_transformer": config.model.resolution_transformer,
            "model_window_transformer": config.model.window_transformer,
        })
    else:
        wandb_run = None

    # ================= 模型（先在 CPU 上构建 & 替换反卷积），再统一上 GPU =================
    model = ClipMatcher(config)           # 先建在 CPU
    _replace_deconv_in_module(model)      # 替换所有 ConvTranspose2d -> Upsample+Conv2d
    model = model.to(device)              # 统一搬到目标设备

    # ================ 优化器 & 调度器（在 .to(device) 之后创建） ================
    optimizer = train_utils.get_optimizer(config, model)
    schedular = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.train.schedular_warmup_iter,
        num_training_steps=config.train.total_iteration,
    )
    scaler = None  # 若你的 train_epoch 没用 AMP，这个就留空

    # ================ 断点恢复（更健壮） ================
    best_iou, best_prob = 0.0, 0.0
    ep_resume = None
    if getattr(config.train, "resume", False):
        model, optimizer, schedular, scaler, ep_resume, best_iou, best_prob = try_resume_safely(
            model, optimizer, schedular, scaler, output_dir, rank=rank
        )

    # ================= DDP 包装（放在优化器创建之后也 OK） =================
    ddp = False
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        print(f"[DDP] world_size={world_size} local_rank={local_rank}")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,   # 如果确认所有分支都参与梯度，可改为 False 更快
        )
        # 可选：通信压缩，降低带宽占用
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import default as ddp_hooks
            model.register_comm_hook(state=None, hook=ddp_hooks.fp16_compress_hook)
            if rank == 0:
                print("[DDP] register fp16_compress_hook")
        except Exception:
            pass
        ddp = True
    else:
        print("[Single Process] Not using DDP. Launch with torchrun for multi-GPU.")

    # # 打印参数量
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Number of trainable parameters: {count_parameters(model)}")
    # def count_all_parameters(model):
    #     return sum(p.numel() for p in model.parameters())
    # print(f"Number of all parameters: {count_all_parameters(model)}")

    # ================== 数据 ==================
    train_loader, val_loader, train_sampler = build_dataloaders(ddp=ddp)

    start_ep = int(ep_resume) if ep_resume is not None else 0
    end_ep = int(config.train.total_iteration / max(1, len(train_loader))) + 1

    # ================== 训练循环 ==================
    for epoch in range(start_ep, end_ep):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_epoch(
            config=config,
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            schedular=schedular,
            scaler=scaler,
            epoch=epoch,
            output_dir=output_dir,
            device=device,
            rank=local_rank,
            ddp=ddp,
            wandb_run=wandb_run,
        )
        torch.cuda.empty_cache()

        # 只让 rank==0 保存
        if rank == 0:
            train_utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict() if ddp else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedular": schedular.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else {},
                    "best_iou": best_iou,
                    "best_prob": best_prob,
                },
                checkpoint=output_dir,
                filename="cpt_last.pth.tar",
            )
            # if epoch % 10 == 0:
            #     train_utils.save_checkpoint(
            #         {
            #             "epoch": epoch + 1,
            #             "state_dict": model.module.state_dict() if ddp else model.state_dict(),
            #             "optimizer": optimizer.state_dict(),
            #             "schedular": schedular.state_dict(),
            #             "scaler": scaler.state_dict() if scaler is not None else {},
            #         },
            #         checkpoint=output_dir,
            #         filename=f"cpt_{epoch+1}.pth.tar",
            #     )
            # 分段保存
            if epoch in range(50, 400):
                train_utils.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.module.state_dict() if ddp else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "schedular": schedular.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else {},
                    },
                    checkpoint=output_dir,
                    filename=f"cpt_{epoch+1}.pth.tar",
                )

        # 验证（降低频次避免刷屏；mask 警告已屏蔽）
        # if epoch % 10 == 0:
        #     if rank == 0:
        #         print("Doing validation...")
        #     iou, prob = validate(
        #         config=config,
        #         loader=val_loader,
        #         model=model,
        #         epoch=epoch,
        #         output_dir=output_dir,
        #         device=device,
        #         rank=local_rank,
        #         ddp=ddp,
        #         wandb_run=wandb_run,
        #     )
        #     torch.cuda.empty_cache()

            # updated = False
            # if iou > best_iou:
            #     best_iou = iou
            #     updated = True
            #     if rank == 0:
            #         train_utils.save_checkpoint(
            #             {
            #                 "epoch": epoch + 1,
            #                 "state_dict": model.module.state_dict() if ddp else model.state_dict(),
            #                 "optimizer": optimizer.state_dict(),
            #                 "schedular": schedular.state_dict(),
            #                 "scaler": scaler.state_dict() if scaler is not None else {},
            #                 "best_iou": best_iou,
            #                 "best_prob": best_prob,
            #             },
            #             checkpoint=output_dir,
            #             filename="cpt_best_iou.pth.tar",
            #         )
            # if prob > best_prob:
            #     best_prob = prob
            #     updated = True
            #     if rank == 0:
            #         train_utils.save_checkpoint(
            #             {
            #                 "epoch": epoch + 1,
            #                 "state_dict": model.module.state_dict() if ddp else model.state_dict(),
            #                 "optimizer": optimizer.state_dict(),
            #                 "schedular": schedular.state_dict(),
            #                 "scaler": scaler.state_dict() if scaler is not None else {},
            #                 "best_iou": best_iou,
            #                 "best_prob": best_prob,
            #             },
            #             checkpoint=output_dir,
            #             filename="cpt_best_prob.pth.tar",
            #         )

            # if rank == 0:
            #     logger.info(
            #         f"Rank {local_rank}, best iou: {best_iou} (current {iou}), "
            #         f"best probability accuracy: {best_prob} (current {prob})"
            #     )

        if ddp:
            dist.barrier()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
