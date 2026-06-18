import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from .backbone import _3DBackbone
from model.transformer import Block
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

# ---------- 3D Anchor 设置 ----------
x_center_num = 16
y_center_num = 16
z_center_num = 16
space_range = [[0., -2.0, -1.0], [10, 2.0, 1.0]]  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]


def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dino':
        assert type in ['vitb8', 'vitb16', 'vits8', 'vits16']
        backbone = torch.hub.load('facebookresearch/dino:main', f'dino_{type}')
        down_rate = int(type.replace('vitb','').replace('vits',''))
        backbone_dim = 768
        if type == 'vitb16' and config.model.bakcbone_use_mae_weight:
            mae_weight = torch.load('/vision/hwjiang/episodic-memory/VQ2D/checkpoint/mae_pretrain_vit_base.pth')['model']
            backbone.load_state_dict(mae_weight)
    elif name == 'dinov2':
        assert type in ['vits14','vitb14','vitl14','vitg14']
        backbone = torch.hub.load('dinov2', f'dinov2_{type}', pretrained=False, source='local')
        state_dict = torch.load('model/dinov2_vitb14_pretrain.pth')
        backbone.load_state_dict(state_dict)
        down_rate = 14
        backbone_dim = 768 if type != 'vits14' else 384
    elif name == 'mae':
        backbone = vit_base_patch16()
        cpt = torch.load('/vision/hwjiang/download/model_weight/mae_pretrain_vit_base.pth')['model']
        backbone.load_state_dict(cpt, strict=False)
        down_rate = 16
        backbone_dim = 768
    else:
        raise ValueError(f'Unknown backbone: {name}')
    return backbone, down_rate, backbone_dim


class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name
        
        # 3d backbone
        self.backbone3d = _3DBackbone(d_axis='x')


        self.query_size = config.dataset.query_size
        self.clip_size_fine = config.dataset.clip_size_fine
        self.clip_size_coarse = config.dataset.clip_size_coarse

        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        self.type_transformer = config.model.type_transformer
        assert self.type_transformer in ['local', 'global']
        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer
        self.resolution_anchor_feat = config.model.resolution_anchor_feat

        # 相机参数
        if hasattr(config, 'camera') and hasattr(config.camera, 'K') and hasattr(config.camera, 'E'):
            self.K = torch.tensor(config.camera.K, dtype=torch.float32)
            self.E = torch.tensor(config.camera.E, dtype=torch.float32)
        else:
            self.K = torch.tensor([[1044.61,0.0,642.33],[0.0,1046.04,367.719],[0.0,0.0,1.0]], dtype=torch.float32)
            self.E = torch.tensor([[-0.019445,-0.999799,-0.00484243,0.0308751],
                                   [-0.0130265,0.00509626,-0.999903,-0.326309],
                                   [0.999617,-0.0263807,-0.0130804,-0.017514],
                                   [0.0,0.0,0.0,1.0]], dtype=torch.float32)

        # 原始图像尺寸
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'origin_h') and hasattr(config.dataset, 'origin_w'):
            self.origin_h = int(config.dataset.origin_h)
            self.origin_w = int(config.dataset.origin_w)
        else:
            self.origin_h, self.origin_w = 720, 1280

        # 3D anchors
        self.center_point = generate_center_on_regions(space_range=space_range,
                                                        num_regions=[x_center_num, y_center_num, z_center_num])

        # query down heads
        # self.query_down_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
        #         nn.BatchNorm2d(self.backbone_dim),
        #         nn.LeakyReLU(inplace=True),
        #     ) for _ in range(int(math.log2(self.query_feat_size)))
        # ])

        # reduce
        self.reduce = nn.Sequential(
            nn.Conv3d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
        )

        # CQ cross-attn
        self.CQ_corr_transformer = nn.ModuleList([
            torch.nn.TransformerDecoderLayer(
                d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0, activation='gelu', batch_first=True
            )
        ])

        # 下采样直到 transformer 分辨率
        self.num_head_layers = int(math.log2(self.clip_feat_size_coarse))
        self.down_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True),
            ) for _ in range(self.num_head_layers - 1)
        ])

        # 4D PE
        self.pe_4d = positionalencoding4d(
            d_model=256,
            height=self.resolution_transformer,
            width=self.resolution_transformer,
            depth=self.resolution_transformer,
            time=config.dataset.clip_num_frames,
            type=config.model.pe_transformer
        ).unsqueeze(0)
        self.pe_4d = nn.parameter.Parameter(self.pe_4d)

        # 时空 transformer
        self.num_transformer = config.model.num_transformer
        self.feat_corr_transformer = nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0, activation='gelu', batch_first=True
            ) for _ in range(self.num_transformer)
        ])



        # === 关键：bool 掩码 buffer（随设备、形状、t 缓存）===
        self.temporal_mask = None
        self._mask_cached_len = 0
        self._mask_cached_t = 0
        self._mask_logged = False  # 只打印一次调试信息

        # 3D head
        self.head = Head3D(in_dim=256, in_res=self.resolution_transformer)#, out_res=self.resolution_anchor_feat)

    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  # [b,h*w,c]
            h = int(h_origin / self.backbone.patch_embed.patch_size)
            w = int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2)
            return (out, h, w) if return_h_w else out
        elif self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h = int(h_origin / self.backbone.patch_embed.patch_size[0])
            w = int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2)
            out = F.interpolate(out, size=(16, 16), mode='bilinear')
            return (out, h, w) if return_h_w else out
        elif self.backbone_name == 'mae':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.forward_features(x)  # [b,1+h*w,c]
            h = int(h_origin / self.backbone.patch_embed.patch_size[0])
            w = int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out[:, 1:].reshape(b, h, w, dim).permute(0, 3, 1, 2)
            out = F.interpolate(out, size=(16, 16), mode='bilinear')
            return (out, h, w) if return_h_w else out
        else:
            raise ValueError(f'Unknown backbone: {self.backbone_name}')

    def replicate_for_hnm(self, query_feat, clip_feat):
        b = query_feat.shape[0]
        bt = clip_feat.shape[0]
        t = bt // b
        clip_feat = rearrange(clip_feat, '(b t) c h w -> b t c h w', b=b, t=t)
        new_clip_feat, new_query_feat = [], []
        for i in range(b):
            for j in range(b):
                new_clip_feat.append(clip_feat[i])
                new_query_feat.append(query_feat[j])
        new_clip_feat = torch.stack(new_clip_feat)      # [b^2,t,c,h,w]
        new_query_feat = torch.stack(new_query_feat)    # [b^2,c,h,w]
        new_clip_feat = rearrange(new_clip_feat, 'b t c h w -> (b t) c h w')
        return new_clip_feat, new_query_feat

    # 3D -> 2D（padding 方形后按 max_size 归一化）
    def bbox3d_to_normbbox_2d(self, bbox_3d, K, E, origin_h, origin_w):
        verts = compute_3d_box_vertices(bbox_3d[..., :3], bbox_3d[..., 3:6], bbox_3d[..., 6:9])  # [b,t,N,8,3]
        verts_2d = project_3d_to_2d(verts, K, E)  # [b,t,N,8,2]
        max_size = float(max(origin_h, origin_w))
        min_size = float(min(origin_h, origin_w))
        pad = (max_size - min_size) / 2.0
        if origin_h < origin_w:
            verts_2d[..., 1] = verts_2d[..., 1] + pad
        else:
            verts_2d[..., 0] = verts_2d[..., 0] + pad
        x_min = verts_2d[..., 0].amin(dim=-1)
        y_min = verts_2d[..., 1].amin(dim=-1)
        x_max = verts_2d[..., 0].amax(dim=-1)
        y_max = verts_2d[..., 1].amax(dim=-1)
        xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        xyxy = (xyxy / max_size).clamp_(0.0, 1.0)
        return xyxy

    def forward(self, clip_img, query_img, query_img_bbox,
                      clip_pcd, query_pcd, query_pcd_bbox,
                      training=False, fix_backbone=True):        
        b, t = clip_img.shape[:2]
        clip_img = rearrange(clip_img, 'b t c h w -> (b t) c h w')
        clip_pcd = rearrange(clip_pcd, 'b t n c -> (b t) n c')

        if fix_backbone:
            with torch.no_grad():
                query_feat = self.extract_feature(query_img)
                clip_feat  = self.extract_feature(clip_img)
                query_feat3d = self.backbone3d(query_pcd)  #[b, c, d, h, w]
                clip_feat3d = self.backbone3d(clip_pcd)      # (b t) c d h w
        else:
            query_feat = self.extract_feature(query_img)
            clip_feat  = self.extract_feature(clip_img)
            query_feat3d = self.backbone3d(query_pcd)      #[b, c, d, h, w]
            clip_feat3d = self.backbone3d(clip_pcd)      # (b t) c d h w

        # fusion
        ## img -> to [-1, -1, D, h, w]
        _, _, D, h, w = clip_feat3d.shape
        clip_feat = clip_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)
        clip_feat = clip_feat + clip_feat3d # [(b t) c d h w]
        c, d, h, w = clip_feat.shape[-4:]

        # 上游 ROI
        idx_tensor = torch.arange(b, device=clip_img.device).float().view(-1, 1)
        query_frame_bbox = dataset_utils.recover_bbox(query_img_bbox, h, w)
        roi_bbox = torch.cat([idx_tensor, query_frame_bbox], dim=1)
        query_feat = torchvision.ops.roi_align(query_feat, roi_bbox, (h, w))

        # 3d ROI
        idx_tensor = torch.arange(b, device=clip_pcd.device).float().view(-1, 1)
        query_pcd_vert = compute_3d_box_vertices_withlist(query_pcd_bbox)  #[b,8,3]
        query_pcd_bbox_6d = point_to_xxyyzz(query_pcd_vert)  #[b,6]
        roi_bbox = torch.cat([idx_tensor, query_pcd_bbox_6d], dim=1) #[b,7]
        query_feat3d = ROI_3d(query_feat3d, roi_bbox, (D,h,w)) #[b, c, d, h, w]

        query_feat = query_feat.unsqueeze(2).repeat(1, 1, D, 1, 1) #[b, c, d, h, w]
        query_feat = query_feat + query_feat3d  #[b, c, d, h, w]

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
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
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
                    b=b, t=t, d=self.resolution_transformer, h=self.resolution_transformer, w=self.resolution_transformer
                )
                break

        # 3D 预测
        # refine center anchor
        center_point = self.center_point.to(clip_feat.device)                   #[n,3]
        center_point = center_point.reshape(1,1,-1,3)                           #[1,1,n,3]

        bbox_refine, prob = self.head(clip_feat)                                #[b*t,d*h*w*n,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  #[b,t,N,9], in xyhw frormulation
        prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                #[b,t,n,1]

        center = bbox_refine[..., 0:3]
        size = F.softplus(bbox_refine[..., 3:6]) / 3.0                           #[b,t,n,3]
        rot = bbox_refine[..., 6:9]
        center = center + center_point

        bbox = torch.cat([center, size, rot], dim=-1)                              #[b,t,n,9]

        return {
            'center': center,           #[b,t,n,3]
            'size': size,                   #[b,t,n,3]
            'rot' : rot,             #[b,t,n,3]
            'bbox': bbox,               #[b,t,n,9]
            'prob': prob.squeeze(-1),   #[b,t,n]
            'center_point': center_point,      #[1,1,n,3]
        }

    def get_mask(self, src, t):
        """
        src: [b, t*d*h*w, c]
        t: number of frames
        """
        if not torch.is_tensor(self.temporal_mask):
            size = src.shape[1] // t
            tdhw = src.shape[1]
            mask = torch.ones(tdhw, tdhw).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*size)
                max_idx = min(tdhw, (i+window_size+1)*size)
                mask[i*size: (i+1)*size, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask


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