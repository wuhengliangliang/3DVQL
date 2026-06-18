# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange

from model.transformer import Block  # 若未用可保留导入以兼容你工程
from utils.model_utils import (
    PositionalEncoding1D,
    positionalencoding1d,
    positionalencoding2d,
    positionalencoding3d,
    positionalencoding4d,
    BasicBlock_Conv2D,
    BasicBlock_Conv3D,
    BasicBlock_MLP,
)
from utils.anchor_utils import generate_center_on_regions
from dataset import dataset_utils
from model.mae import vit_base_patch16
from utils.mathConvert import (
    compute_3d_box_vertices_withlist,
    compute_3d_box_vertices,
    compute_rotation_matrix_from_directions,
    project_3d_to_2d,
    point_to_xxyyzz,
    ROI_3d
)
# =============== 3D center / 空间范围配置（与你现有工程保持一致） ===============
x_center_num = 32
y_center_num = 16
z_center_num = 8
space_range = [[0.0, -2.0, -1.0], [10.0, 2.0, 1.0]]  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]

# =============== 导入你的 DGCNN backbone（放在同目录的 backbone 包里） ===============
# 需要你提供 .backbone.DGCNNBackbone 实现（返回 [B,emb,N] 或 {'feat': [B,emb,N]}）
from .backbone import  _3DBackbone


# ======================== Backbone 构建（支持离线） ========================
def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dino':
        assert type in ['vitb8', 'vitb16', 'vits8', 'vits16']
        backbone = torch.hub.load('facebookresearch/dino:main', f'dino_{type}')
        down_rate = int(type.replace('vitb', '').replace('vits', ''))
        backbone_dim = 768
        if type == 'vitb16' and getattr(config.model, 'bakcbone_use_mae_weight', False):
            # 仅当你本地有该权重
            mae_path = getattr(config.model, 'mae_weight_path',
                               '/vision/hwjiang/episodic-memory/VQ2D/checkpoint/mae_pretrain_vit_base.pth')
            if os.path.exists(mae_path):
                mae_weight = torch.load(mae_path)['model']
                backbone.load_state_dict(mae_weight, strict=False)
    elif name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        # 强制本地加载，避免联网
        try:
            backbone = torch.hub.load('dinov2', f'dinov2_{type}', pretrained=False, source='local')
        except Exception:
            # 兼容某些环境 hub 源名为 'facebookresearch/dinov2' 且无法联网
            # 这里仅创建一个占位错误以便用户知晓
            raise RuntimeError(
                "Unable to load dinov2 backbone via torch.hub(local). "
                "Please ensure dinov2 is available locally for torch.hub."
            )
        # 加载本地权重（可从 config 指定）
        weight_path = getattr(config.model, 'dinov2_weight_path', 'model/dinov2_vitb14_pretrain.pth')
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu')
            backbone.load_state_dict(state_dict, strict=False)
        down_rate = 14
        backbone_dim = 768 if type != 'vits14' else 384
    elif name == 'mae':
        backbone = vit_base_patch16()
        cpt_path = getattr(config.model, 'mae_weight_path',
                           '/vision/hwjiang/download/model_weight/mae_pretrain_vit_base.pth')
        if os.path.exists(cpt_path):
            cpt = torch.load(cpt_path, map_location='cpu')['model']
            backbone.load_state_dict(cpt, strict=False)
        down_rate = 16
        backbone_dim = 768
    else:
        raise ValueError(f'Unknown backbone: {name}')
    return backbone, down_rate, backbone_dim

# ======================== 小模块 ========================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d)
        self.bn   = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AttentionFusion(nn.Module):
    def __init__(self, channels, num_heads=8, axis=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.axis = axis
        
        # 投影层
        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.k_proj = nn.Conv3d(channels, channels, 1)
        self.v_proj = nn.Conv3d(channels, channels, 1)

        # 注意力机制（只关注深度维度）
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 输出层
        self.out_proj = nn.Conv3d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, pcd_feat, img_feat):
        """
        pcd_feat: 点云特征 [B, C, D, H, W]
        img_feat: 图像特征 [B, C, D, H, W]
        """
        B, C, D, H, W = pcd_feat.shape
        
        # 投影到query/key/value
        q = self.q_proj(pcd_feat)  # [B, C, D, H, W]
        k = self.k_proj(img_feat)  # [B, C, D, H, W]
        v = self.v_proj(img_feat)  # [B, C, D, H, W]
        
        # 将空间维度展平，保留axis维度
        q_seq, k_seq, v_seq = None, None, None
        if self.axis == "D":
            q_seq = rearrange(q, 'B C D H W -> (B H W) D C')
            k_seq = rearrange(k, 'B C D H W -> (B H W) D C')
            v_seq = rearrange(v, 'B C D H W -> (B H W) D C')
        elif self.axis == "H":
            q_seq = rearrange(q, 'B C D H W -> (B D W) H C')
            k_seq = rearrange(k, 'B C D H W -> (B D W) H C')
            v_seq = rearrange(v, 'B C D H W -> (B D W) H C')
        elif self.axis == "W":
            q_seq = rearrange(q, 'B C D H W -> (B D H) W C')
            k_seq = rearrange(k, 'B C D H W -> (B D H) W C')
            v_seq = rearrange(v, 'B C D H W -> (B D H) W C')
        elif self.axis == "all":
            q_seq = rearrange(q, 'B C D H W -> (B) (D H W) C')
            k_seq = rearrange(k, 'B C D H W -> (B) (D H W) C')
            v_seq = rearrange(v, 'B C D H W -> (B) (D H W) C')

        # 注意力计算（只在深度维度）
        attn_output, _ = self.attention(
            query=q_seq,
            key=k_seq,
            value=v_seq
        ) # [(B H W), D, C] or [(B D W), H, C] or [(B D H), W, C]
        
        # 残差连接
        attn_output = attn_output + q_seq
        
        # 层归一化
        attn_output = self.norm(attn_output)
        
        # 恢复空间维度
        # attn_output = rearrange(attn_output, '(B H W) D C -> B C D H W', B=B, H=H, W=W)
        if self.axis == "D":
            attn_output = rearrange(attn_output, '(B H W) D C -> B C D H W', B=B, H=H, W=W)
        elif self.axis == "H":
            attn_output = rearrange(attn_output, '(B D W) H C -> B C D H W', B=B, D=D, W=W)
        elif self.axis == "W":
            attn_output = rearrange(attn_output, '(B D H) W C -> B C D H W', B=B, D=D, H=H)
        elif self.axis == "all":
            attn_output = rearrange(attn_output, '(B) (D H W) C -> B C D H W', B=B, D=D, H=H, W=W)
        # 输出投影
        output = self.out_proj(attn_output)
        return output
# ======================== 3D Head（anchor refine） ========================
# 3d Head [b, c, d, h, w] -> [b, n, 9] center point way
class Head3D(nn.Module):
    def __init__(self, in_dim=256, in_res=8, out_res=16, num = 1):
        super(Head3D, self).__init__()

        self.in_dim = in_dim
        self.num = num
        self.num_up_layers = int(math.log2(out_res // in_res))
        self.num_layers = 3

        if self.num_up_layers > 0:
            self.up_convs = []
            for _ in range(self.num_up_layers):
                self.up_convs.append(torch.nn.ConvTranspose3d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
            self.up_convs = nn.Sequential(*self.up_convs)

        self.in_conv = BasicBlock_Conv3D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv3D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv3D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv3d(in_dim, num * 9, kernel_size=3, padding=1)
        self.classification_head = nn.Conv3d(in_dim, num * 1, kernel_size=3, padding=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv3d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        if self.num_up_layers > 0:
            x = self.up_convs(x.clone())  #Create a copy of the upsampling operation

        B, c, d, h, w = x.shape

        #Use the clone() method to create a copy of the input to avoid in-place modifications
        feat_reg, feat_cls = self.in_conv(x.clone()).split([c, c], dim=1)

        #Use dropout operation here
        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        feat_reg = self.regression_conv(feat_reg)
        feat_cls = self.classification_conv(feat_cls)

        out_reg = self.regression_head(feat_reg)
        out_cls = self.classification_head(feat_cls)

        out_reg = rearrange(out_reg, 'B (num c) d h w -> B (d h w num) c', d=d, h=h, w=w, num=self.num, c=9)
        out_cls = rearrange(out_cls, 'B (num c) d h w -> B (d h w num) c', d=d, h=h, w=w, num=self.num, c=1)

        return out_reg, out_cls # [b, n, 9], [b, n, 1]

# ======================== 主模型（VQLoC 2D + DGCNN 融合 + 3D Head） ========================
class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # camera（可选，仅在你做 3D->2D 可视化时有用）
        if hasattr(config, 'camera') and hasattr(config.camera, 'K') and hasattr(config.camera, 'E'):
            self.K = torch.tensor(config.camera.K, dtype=torch.float32)
            self.E = torch.tensor(config.camera.E, dtype=torch.float32)
        else:
            self.K = torch.tensor([[1044.61,0.0,642.33],[0.0,1046.04,367.719],[0.0,0.0,1.0]], dtype=torch.float32)
            self.E = torch.tensor([[-0.019445,-0.999799,-0.00484243,0.0308751],
                                   [-0.0130265,0.00509626,-0.999903,-0.326309],
                                   [0.999617,-0.0263807,-0.0130804,-0.017514],
                                   [0.0,0.0,0.0,1.0]], dtype=torch.float32)

        # 原图尺寸（仅在投影/可视化时用）
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'origin_h') and hasattr(config.dataset, 'origin_w'):
            self.origin_h = int(config.dataset.origin_h)
            self.origin_w = int(config.dataset.origin_w)
        else:
            self.origin_h, self.origin_w = 720, 1280

        # 2D backbone
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name

        # 尺度配置
        self.query_size = config.dataset.query_size
        self.clip_size_fine = config.dataset.clip_size_fine
        self.clip_size_coarse = config.dataset.clip_size_coarse
        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        # Transformer 相关配置
        self.type_transformer = config.model.type_transformer
        assert self.type_transformer in ['local', 'global']
        self.window_transformer = int(config.model.window_transformer)
        self.resolution_transformer = int(config.model.resolution_transformer)
        self.resolution_anchor_feat = int(config.model.resolution_anchor_feat)

        # 3D anchors（静态，前向时搬到 device）
        self.center_point = generate_center_on_regions(space_range=space_range,
                                                        num_regions=[x_center_num, y_center_num, z_center_num])

        # Query 下采样（对 2D backbone 输出：如果需要进一步下采样到与 clip 对齐）
        self.query_down_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
                nn.BatchNorm3d(self.backbone_dim),
                nn.LeakyReLU(inplace=True),
            )
            for _ in range(max(0, int(math.log2(max(1, self.query_feat_size)))))
        ])

        # 融合前后的通道对齐（最终都降到 256）
        self.reduce = nn.Sequential(
            nn.Conv3d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
        )

        # 2D-Query 与 Clip 的 Cross-Attention（仅 1 层，保持与 VQLoC 类似）
        self.CQ_corr_transformer = nn.ModuleList([
            torch.nn.TransformerDecoderLayer(
                d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0,
                activation='gelu', batch_first=True
            )
        ])

        # 下采样直到时空 Transformer 分辨率
        self.num_head_layers = int(math.log2(max(1, self.clip_feat_size_coarse)))
        self.down_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True),
            )
            for _ in range(max(0, self.num_head_layers - 1))
        ])

        # 4D PE（T,D,H,W）
        self.pe_4d = positionalencoding4d(
            d_model=256,
            height=int(self.resolution_transformer / 2),
            width=self.resolution_transformer,
            depth=self.resolution_transformer * 2,
            time=config.dataset.clip_num_frames,
            type=config.model.pe_transformer
        ).unsqueeze(0)
        self.pe_4d = nn.parameter.Parameter(self.pe_4d)

        # 时空 Transformer
        self.num_transformer = int(config.model.num_transformer)
        self.feat_corr_transformer = nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0,
                activation='gelu', batch_first=True
            )
            for _ in range(self.num_transformer)
        ])

        # mask 缓存（bool）
        self.temporal_mask = None
        self._mask_cached_len = 0
        self._mask_cached_t = 0
        self._mask_logged = False  # 只打印一次调试信息

        # 3D head
        self.head = Head3D(
            in_dim=256,
            in_res=self.resolution_transformer,
        )

        # ------------------ 3D 点云 backbone 与 2D-3D 融合 ------------------
        # 你的 DGCNN 设置：需要 config.model.backbone3d.emb_dims
        self.backbone3d = _3DBackbone(out_channels=768)

        # fusion attension :pcd_feat [B,T,D,H,W], img_feat [B,T,D,H,W]
        self.D_fusion = AttentionFusion(
            channels=768,
            num_heads=8,
            axis="D"
        )
        self.D_pe = positionalencoding3d(
            d_model=768,
            height=x_center_num,
            width=y_center_num,
            depth=z_center_num,
            type=config.model.pe_transformer
        ).unsqueeze(0) # [1, C, D, H, W]
        self.D_pe = self.D_pe.view(1, 768, x_center_num, z_center_num, y_center_num)
        self.D_pe = nn.parameter.Parameter(self.D_pe)

        # self.H_fusion = AttentionFusion(
        #     channels=768,
        #     num_heads=8,
        #     axis="H"
        # )
        # self.W_fusion = AttentionFusion(
        #     channels=768,
        #     num_heads=8,
        #     axis="W"
        # )
        # self.fusion = AttentionFusion(
        #     channels=768,
        #     num_heads=8,
        #     axis="all"
        # )
        # 时空 Transformer for fusion
        # self.fusion_pe_img = positionalencoding3d(
        #     d_model=768,
        #     height=self.resolution_anchor_feat,
        #     width=self.resolution_anchor_feat,
        #     depth=self.resolution_anchor_feat,
        #     type=config.model.pe_transformer
        # ).unsqueeze(0)
        # self.fusion_pe_img = nn.parameter.Parameter(self.fusion_pe_img)
        # self.fusion_pe_pcd = positionalencoding3d(
        #     d_model=768,
        #     height=self.resolution_anchor_feat,
        #     width=self.resolution_anchor_feat,
        #     depth=self.resolution_anchor_feat,
        #     type=config.model.pe_transformer
        # ).unsqueeze(0)
        # self.fusion_pe_pcd = nn.parameter.Parameter(self.fusion_pe_pcd)

        # self.fusion_transformer = nn.ModuleList([
        # AttentionFusion(
        #     channels=768,
        #     num_heads=8,
        #     axis="D"
        # )
        #     for _ in range(self.num_transformer)
        # ]) 

        # self.self_attention = nn.ModuleList([
        #     AttentionFusion(
        #         channels=768,
        #         num_heads=8,
        #         axis="D"
        #     )
        #     for _ in range(self.num_transformer)
        # ])

    # ------------------ 2D / 3D feature 提取 ------------------
    def project_imgfeat_to_3dfeat(self, pcd_feat, img_feat):
        # pcd_feat: [B, C, D, H, W] with x->D, y->W, z->H
        B, _, D, H, W = pcd_feat.shape
        b, C, H0, W0 = img_feat.shape
        device = img_feat.device

        # 统一坐标约定：x->D, y->W, z->H
        x_lin = torch.linspace(space_range[0][0], space_range[1][0], steps=D, device=device)  # x over D
        y_lin = torch.linspace(space_range[0][1], space_range[1][1], steps=W, device=device)  # y over W
        z_lin = torch.linspace(space_range[0][2], space_range[1][2], steps=H, device=device)  # z over H

        # 生成 [D,H,W] 的栅格，其中 X 对应 x, Y 对应 y, Z 对应 z
        # 注意：为了得到 [D,H,W]，用 (x, z, y) 顺序调用 meshgrid，再按 (X, Y, Z) 组装
        X, Z, Y = torch.meshgrid(x_lin, z_lin, y_lin, indexing='ij')  # shapes: [D,H,W]
        grid_3d = torch.stack([X, Y, Z], dim=-1).view(-1, 3)          # (x,y,z) 扁平化
    
        # 投影到图像平面
        uv = project_3d_to_2d(
            points_3d=grid_3d,
            intrinsic=self.K.to(device),
            extrinsic=self.E.to(device)
        )  # [(D*H*W), 2]

        # 填充到方形再缩放到特征图分辨率
        base = max(self.origin_h, self.origin_w)
        if self.origin_h > self.origin_w:
            uv[:, 0] = uv[:, 0] + (self.origin_h - self.origin_w) * 0.5
        elif self.origin_w > self.origin_h:
            uv[:, 1] = uv[:, 1] + (self.origin_w - self.origin_h) * 0.5
        sx, sy = W0 / base, H0 / base
        uv[:, 0] *= sx
        uv[:, 1] *= sy

        # 归一化到 [-1,1]，保持与 [D,H,W] 展平顺序一致
        grid_2d = torch.empty((D, H, W, 2), device=device, dtype=img_feat.dtype)
        grid_2d[..., 0] = (uv[:, 0].view(D, H, W) / max(W0 - 1, 1)) * 2 - 1  # u -> width (W)
        grid_2d[..., 1] = (uv[:, 1].view(D, H, W) / max(H0 - 1, 1)) * 2 - 1  # v -> height (H)

        # 逐深度切片做双线性采样
        grid_bt = grid_2d.unsqueeze(0).repeat(B, 1, 1, 1, 1).view(B * D, H, W, 2)
        img_bt  = img_feat.unsqueeze(1).repeat(1, D, 1, 1, 1).view(B * D, C, H0, W0)
        out_bt  = F.grid_sample(img_bt, grid_bt, mode='bilinear', align_corners=True)
        out     = out_bt.view(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,D,H,W]
        return out


    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dino':
            b, _, h0, w0 = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]  # [b, 1+h*w, c] or [b,h*w+1,c]
            out = out[:, 1:, :]                                     # remove [CLS]
            h = int(h0 / self.backbone.patch_embed.patch_size)
            w = int(w0 / self.backbone.patch_embed.patch_size)
            c = out.shape[-1]
            out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        elif self.backbone_name == 'dinov2':
            b, _, h0, w0 = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h = int(h0 / self.backbone.patch_embed.patch_size[0])
            w = int(w0 / self.backbone.patch_embed.patch_size[1])
            c = out.shape[-1]
            out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
            out = F.interpolate(out, size=(8, 16), mode='bilinear')
        elif self.backbone_name == 'mae':
            b, _, h0, w0 = x.shape
            out = self.backbone.forward_features(x)  # [b, 1+h*w, c]
            h = int(h0 / self.backbone.patch_embed.patch_size[0])
            w = int(w0 / self.backbone.patch_embed.patch_size[1])
            c = out.shape[-1]
            out = out[:, 1:].reshape(b, h, w, c).permute(0, 3, 1, 2)
            # 与你工程一致固定到 16x16
            out = F.interpolate(out, size=(16, 16), mode='bilinear')
        else:
            raise ValueError(f'Unknown backbone: {self.backbone_name}')
        return (out, h, w) if return_h_w else out

    def fusion_feature(self, img_feat, pcd_feat):
        """
        img_feat: [B, C, H, W]
        pcd_feat: [B, C, D, H, W]
        返回 fused_feat: [B, C, D, H, W]
        """
        b, c, d, h, w = pcd_feat.shape
        # img_feat = img_feat.unsqueeze(2).repeat(1, 1, d, 1, 1)  #[b, c, d, h, w]
        img_feat = self.project_imgfeat_to_3dfeat(
            pcd_feat=pcd_feat,
            img_feat=img_feat
        )  #[b, c, d, h, w]
        # add PE
        img_feat = img_feat + self.D_pe.repeat(b, 1, 1, 1, 1)
        pcd_feat = pcd_feat + self.D_pe.repeat(b, 1, 1, 1, 1)

        fused_feat_D = self.D_fusion(
            pcd_feat=pcd_feat,
            img_feat=img_feat
        )  #[b, c, d, h, w]
        # fused_feat_H = self.H_fusion(
        #     pcd_feat=pcd_feat,
        #     img_feat=img_feat
        # )  #[b, c, d, h, w]
        # fused_feat_W = self.W_fusion(
        #     pcd_feat=pcd_feat,
        #     img_feat=img_feat
        # )  #[b, c, d, h, w]
        # fused_feat = 0.25*fused_feat_W + 0.25*fused_feat_H + 0.5*fused_feat_D
        # fused_feat = self.fusion(
        #     pcd_feat=pcd_feat,
        #     img_feat=img_feat
        # )  #[b, c, d, h, w]
        fused_feat = fused_feat_D
        
        # 时空 Transformer 融合
        # img_feat_seq = rearrange(img_feat, 'b c d h w -> b (d h w) c')
        # pcd_feat_seq = rearrange(pcd_feat, 'b c d h w -> b (d h w) c')
        # img_feat_seq = img_feat_seq + self.fusion_pe_img
        # pcd_feat_seq = pcd_feat_seq + self.fusion_pe_pcd
        # img_feat = rearrange(img_feat_seq, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        # pcd_feat = rearrange(pcd_feat_seq, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

        # fusion_feat = pcd_feat
        # for layer in self.fusion_transformer:
        #     fusion_feat = layer(
        #         fusion_feat,
        #         img_feat
        #     )
        # selfattention
        # for layer in self.self_attention:
        #     fusion_feat = layer(
        #         fusion_feat,
        #         fusion_feat
        #     )

        return fused_feat
    # ------------------ HNM 复制（与 VQLoC 一致） ------------------
    def replicate_for_hnm(self, query_feat, clip_feat):
        """
        query_feat: [B, C, H, W]
        clip_feat:  [(B*T), C, H, W]
        返回 new_clip_feat, new_query_feat
        """
        b = query_feat.shape[0]
        bt = clip_feat.shape[0]
        t = bt // b
        clip_feat_5d = rearrange(clip_feat, '(b t) c h w -> b t c h w', b=b, t=t)

        combo_clip, combo_query = [], []
        for i in range(b):
            for j in range(b):
                combo_clip.append(clip_feat_5d[i])
                combo_query.append(query_feat[j])
        new_clip  = torch.stack(combo_clip)               # [B^2, T, C, H, W]
        new_query = torch.stack(combo_query)              # [B^2, C, H, W]
        new_clip  = rearrange(new_clip, 'bb t c h w -> (bb t) c h w')
        return new_clip, new_query

    # ------------------ 时序窗口 Mask（bool 缓存） ------------------
    def get_mask(self, src, t):
        """
        src: [B, TDHW, C]（其中 TDHW = T*D*H*W）
        t:   序列长度
        返回: [TDHW, TDHW] bool mask，True=屏蔽，False=允许
        """
        tdhw = src.shape[1]
        size = tdhw // max(1, t)

        need_rebuild = (
            self.temporal_mask is None or
            self.temporal_mask.numel() == 0 or
            self.temporal_mask.shape[0] != tdhw or
            self._mask_cached_t != t or
            self.temporal_mask.device != src.device
        )
        if need_rebuild:
            window = self.window_transformer // 2
            mask = torch.ones((tdhw, tdhw), dtype=torch.bool, device=src.device)
            for i in range(t):
                lo = max(0, (i - window) * size)
                hi = min(tdhw, (i + window + 1) * size)
                mask[i*size:(i+1)*size, lo:hi] = False
            self.temporal_mask = mask
            self._mask_cached_len = tdhw
            self._mask_cached_t = t

        # 再保险 dtype/device
        if self.temporal_mask.dtype is not torch.bool:
            self.temporal_mask = self.temporal_mask.to(torch.bool)
        if self.temporal_mask.device != src.device:
            self.temporal_mask = self.temporal_mask.to(src.device)
        return self.temporal_mask

    # ------------------ 前向（含老接口别名） ------------------
    def forward(self,
                clip_img=None,              # [B,T,C,H,W] (new)
                query_img=None,             # [B,C,h2,w2] (new)
                query_img_bbox=None,        # [B,4]       (new)
                clip_pcd=None,              # [B,T,N,3] or None
                query_pcd=None,             # [B,N,3] or None
                query_pcd_bbox=None,        # [B,6] or None
                # ---- VQLoC 老接口别名 ----
                query_frame_bbox=None,      # alias of query_img_bbox (old)
                clip=None,                  # alias of clip_img (old)
                query=None,                 # alias of query_img (old)
                # ---------------------------
                training=False,
                fix_backbone=True,
                **kwargs):
        """
        返回 keys: center / size / rot / bbox / prob / anchor
        """
        b, t = clip_img.shape[:2]
        clip_img = rearrange(clip_img, 'b t c h w -> (b t) c h w')
        clip_pcd = rearrange(clip_pcd, 'b t n c -> (b t) n c')

        if fix_backbone:
            with torch.no_grad():
                query_feat = self.extract_feature(query_img)
                clip_feat  = self.extract_feature(clip_img)
                # query_feat3d = self.backbone3d(query_pcd)  #[b, c, d, h, w]
                # clip_feat3d = self.backbone3d(clip_pcd)      # (b t) c d h w
        else:
            query_feat = self.extract_feature(query_img)
            clip_feat  = self.extract_feature(clip_img)
            # query_feat3d = self.backbone3d(query_pcd)      #[b, c, d, h, w]
            # clip_feat3d = self.backbone3d(clip_pcd)      # (b t) c d h w
        query_feat3d = self.backbone3d(query_pcd)    #[b, c, d, h, w]
        clip_feat3d = self.backbone3d(clip_pcd)      # (b t) c d h w

        # fusion
        ## img -> to [-1, -1, D, h, w]
        _, _, D, h, w = clip_feat3d.shape
        # clip_feat = clip_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)
        # clip_feat = clip_feat + clip_feat3d # [(b t) c d h w]
        # c, d, h, w = clip_feat.shape[-4:]
        ## rearrange to [b t, d h w, c]

        # fusion
        clip_feat = self.fusion_feature(
            img_feat=clip_feat,
            pcd_feat=clip_feat3d
        )
        query_feat = self.fusion_feature(
            img_feat=query_feat,
            pcd_feat=query_feat3d,
        )

        # 上游 ROI
        # idx_tensor = torch.arange(b, device=clip_img.device).float().view(-1, 1)
        # query_frame_bbox = dataset_utils.recover_bbox(query_img_bbox, h, w)
        # roi_bbox = torch.cat([idx_tensor, query_frame_bbox], dim=1)
        # query_feat = torchvision.ops.roi_align(query_feat, roi_bbox, (h, w))

        # 3d ROI
        idx_tensor = torch.arange(b, device=clip_pcd.device).float().view(-1, 1)
        query_pcd_vert = compute_3d_box_vertices_withlist(query_pcd_bbox)  #[b,8,3]
        query_pcd_bbox_6d = point_to_xxyyzz(query_pcd_vert)  #[b,6]
        roi_bbox = torch.cat([idx_tensor, query_pcd_bbox_6d], dim=1) #[b,7]
        query_feat = ROI_3d(query_feat, roi_bbox, (D,h,w)) #[b, c, d, h, w]

        # 降维
        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        if self.config.train.use_hnm and training:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)
            b = b ** 2

        # 空间相关（CQ）
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1, t, 1, 1, 1, 1), 'b t c d h w -> (b t) (d h w) c')
        clip_feat  = rearrange(clip_feat, 'b c d h w -> b (d h w) c')
        for layer in self.CQ_corr_transformer:
            clip_feat = layer(clip_feat, query_feat)
        clip_feat = rearrange(clip_feat, 'b (d h w) c -> b c d h w', h=h, w=w)

        # 时空相关（带局部窗口 mask）
        for head in self.down_heads:
            clip_feat = head(clip_feat)
            if list(clip_feat.shape[-1:]) == [self.resolution_transformer]:
                clip_feat = rearrange(clip_feat, '(b t) c d h w -> b (t d h w) c', b=b) + self.pe_4d

                mask = self.get_mask(clip_feat, t)                               # bool, [TDHW,TDHW]
                mask = mask.to(device=clip_feat.device, dtype=torch.bool)        # 再保险
                if not self._mask_logged:
                    # print(f"[MaskDebug] dtype={mask.dtype}, shape={tuple(mask.shape)}, true_ratio={(mask.float().mean().item()):.4f}")
                    self._mask_logged = True

                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)                  # 只传 bool

                clip_feat = rearrange(
                    clip_feat, 'b (t d h w) c -> (b t) c d h w',
                    b=b, t=t, d=self.resolution_transformer * 2 , h=int(self.resolution_transformer / 2), w=self.resolution_transformer
                )
                break

        # 3D 预测
        # refine center anchor
        center_point = self.center_point.to(clip_feat.device)                   #[n,3]
        center_point = center_point.view(x_center_num, y_center_num, z_center_num, 3) #[x_num,y_num,z_num,3]
        center_point = center_point.permute(0,2,1,3).contiguous()               #[x_num,z_num,y_num,3]
        center_point = center_point.view(-1,3)                                  #[n,3]
        center_point = center_point.reshape(1,1,-1,3)                           #[1,1,n,3]

        bbox_refine, prob = self.head(clip_feat)                                #[b*t,d*h*w*n,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  #[b,t,N,9], in xyhw frormulation
        prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                #[b,t,n,1]

        center = bbox_refine[..., 0:3]
        size = F.softplus(bbox_refine[..., 3:6]) / 3.0                           #[b,t,n,3]
        rot = bbox_refine[..., 6:9]
        center = center + center_point
        # center 重排到 (x,y,z)
        center = center.view(b, t, x_center_num, z_center_num, y_center_num, 3) #[b,t,x_num,z_num,y_num,3]
        center = center.permute(0,1,2,4,3,5).contiguous()               #[b,t,x_num,y_num,z_num,3]
        center = center.view(b, t, -1, 3)                                  #[b,t,n,3]

        bbox = torch.cat([center, size, rot], dim=-1)                              #[b,t,n,9]
        # center 重排到 (x,y,z)
        center_point = self.center_point.to(clip_feat.device)                   #[n,3]
        center_point = center_point.reshape(1,1,-1,3)                           #[1,1,n,3]

        return {
            'center': center,           #[b,t,n,3]
            'size': size,                   #[b,t,n,3]
            'rot' : rot,             #[b,t,n,3]
            'bbox': bbox,               #[b,t,n,9]
            'prob': prob.squeeze(-1),   #[b,t,n]
            'center_point': center_point,      #[1,1,n,3]
        }
