import yaml
import os
import numpy as np
from easydict import EasyDict as edict

config = edict()

# experiment config
config.exp_name = 'vqloc'
config.exp_group = 'baseline'
config.output_dir = '/mnt/data_2/pl/VQLOC3dF/output/VQLOC/train/train/'
config.log_dir = './log'
config.workers = 4
config.print_freq = 100
config.vis_freq = 300
config.eval_vis_freq = 20
config.seed = 42
config.inference_cache_path = ''
config.debug = False

# ========================
# dataset config（补齐root等）
# ========================
config.dataset = edict()
config.dataset.root = '/home/UNT/bf0191/Documents/vq2d'   # ← 补上
config.dataset.name = 'ego4d_vq2d'
config.dataset.name_val = 'ego4d_vq2d'
config.dataset.query_size = 256
config.dataset.clip_size_fine = 256
config.dataset.clip_size_coarse = 256
config.dataset.clip_num_frames = 20
config.dataset.clip_num_frames_val = 20
config.dataset.clip_sampling = 'rand'
config.dataset.clip_reader = 'decord_balance'
config.dataset.clip_reader_val = 'decord_balance'
config.dataset.frame_interval = 5
config.dataset.query_padding = False
config.dataset.query_square = False
config.dataset.padding_value = 'zero'

# 兼容上面配置里用到的预加载/点数
config.dataset.preload_offset = 10
config.dataset.frame_npts = 1024

# =============
# model config
# =============
config.model = edict()
config.model.backbone_name = 'dino'
config.model.backbone_type = 'vitb8'
config.model.bakcbone_use_mae_weight = False
config.model.fix_backbone = True
config.model.num_transformer = 2
config.model.type_transformer = 'global'
config.model.resolution_transformer = 1
config.model.resolution_anchor_feat = 1
config.model.pe_transformer = 'sinusoidal'
config.model.window_transformer = 10
config.model.positive_threshold = 0.2
config.model.positive_topk = 5
config.model.cpt_path = ''
# 与上面保持一致（如用到）
config.model.updatequeries_cpt_path = ''
config.model.trans_dim = 384

# 3D backbone（保持与上面一致）
config.model.backbone3d = edict()
config.model.backbone3d.type = 'DGCNN'
# config.model.backbone3d.out_channels = 128
# config.model.backbone3d.downsample_ratios = [2, 4, 8]
config.model.backbone3d.layers_cfg = [
    {
        'mlps': [0, 64, 64, 128],
        'use_xyz': True,
        'sample_method': 'Range',
        'nsample': 32,
    },
    {
        'mlps': [128, 128, 128, 128],
        'use_xyz': True,
        'sample_method': 'Range',
        'nsample': 32,
    },
    {
        'mlps': [128, 256, 256, 256],
        'use_xyz': True,
        'sample_method': 'Range',
        'nsample': 32,
    },
]

# ===========
# loss config
# ===========
config.loss = edict()
config.loss.weight_bbox = 1.0
config.loss.weight_bbox_center = 1.0
config.loss.weight_bbox_hw = 1.0
config.loss.weight_bbox_size = 1.0      # ← 补上（3D分支要用）
config.loss.weight_bbox_rot = 1.0       # ← 补上（3D分支要用）
config.loss.weight_bbox_ratio = 1.0
config.loss.weight_bbox_distance = 0.3
config.loss.weight_bbox_giou = 0.1
config.loss.weight_rot = 1.0            # ← 上面loss里引用到的键
config.loss.weight_prob = 100.0         # ← 与上面保持一致（原来是1.0）
config.loss.prob_bce_weight = [0.05, 0.95]

# ==============
# training config
# ==============
config.train = edict()
config.train.resume = False
config.train.batch_size = 4
config.train.total_iteration = 5000000
config.train.lr = 0.001
config.train.weight_decay = 0.0001
config.train.schedular_warmup_iter = 1000
config.train.schedualr_milestones = [15000, 30000, 45000]
config.train.schedular_gamma = 0.3
config.train.grad_max = 20.0
config.train.accumulation_step = 1

# 与“上面的代码”保持一致（False/False/True）
config.train.aug_clip = False
config.train.aug_query = False
config.train.aug_clip_iter = 10000
config.train.aug_brightness = 0.2
config.train.aug_contrast = 0.2
config.train.aug_saturation = 0.2
config.train.aug_crop_scale = 0.8
config.train.aug_crop_ratio_min = 0.8
config.train.aug_crop_ratio_max = 1.2
config.train.aug_affine_degree = 90
config.train.aug_affine_translate = 0.1
config.train.aug_affine_scale_min = 0.9
config.train.aug_affine_scale_max = 1.1
config.train.aug_affine_shear_min = -15.0
config.train.aug_affine_shear_max = 15.0
config.train.aug_prob_color = 0.2
config.train.aug_prob_flip = 0.2
config.train.aug_prob_crop = 0.2
config.train.aug_prob_affine = 0.2
config.train.use_hnm = False
config.train.use_query_roi = True        # ← 与上面一致

# =========
# test cfg
# =========
config.test = edict()
config.test.batch_size = 4
config.test.compute_metric = True
config.test.fg_threshold = 0.5

# ==========
# iterations
# ==========
config.iterations = edict()
config.iterations.topK_nums = 3
config.iterations.topK_threshold = 0.7
config.iterations.query_iter = 2

# =========
# rpn cfg
# =========
config.rpn_cfg = edict()
config.rpn_cfg.feat_dim = 128
config.rpn_cfg.n_smp_x = 3
config.rpn_cfg.n_smp_y = 3
config.rpn_cfg.n_smp_z = 5
config.rpn_cfg.n_proposals = 64
config.rpn_cfg.n_proposals_train = 48
config.rpn_cfg.sample_method = 'shrink'
config.rpn_cfg.edge_aggr = {
    'pre_mlps': [129, 128, 128],
    'mlps': [128, 128, 128],
    'use_xyz': True,
    'nsample': 8,
}

# ============
# inference cfg
# ============
config.inference = edict()
config.inference.smoothing_sigma = 5
config.inference.peak_score_threshold = 0.7
config.inference.peak_window_threshold = 0.4

# ======================
# 合并 yaml 的工具函数
# ======================
def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
