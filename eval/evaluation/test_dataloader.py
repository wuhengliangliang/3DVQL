import torch
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import kornia
from einops import rearrange
from dataset.dataset_utils import NORMALIZE_MEAN, NORMALIZE_STD
from dataset.base_dataset import get_bbox_from_data
from dataset import dataset_utils

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_query(config, clip_reader, visual_crop, clip_path):
    """
    Return query tensor: [C, H, W], RGB
    """
    vc_fno = visual_crop["frame_number"]
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]

    if vc_fno >= len(clip_reader):
        print(f"=====> WARNING: Out of range. {clip_path}, Len: {len(clip_reader)}, j: {vc_fno}")

    # 读取帧 (RGB, HWC)
    query = clip_reader.get_batch([vc_fno])[0].numpy()
    if query.shape[:2] != (oheight, owidth):
        query = cv2.resize(query, (owidth, oheight))

    query = Image.fromarray(query)

    # 加载bbox
    bbox = get_bbox_from_data(visual_crop)
    if config.dataset.query_square:
        bbox = dataset_utils.bbox_cv2Totorch(torch.tensor(bbox))
        bbox = dataset_utils.create_square_bbox(bbox, oheight, owidth)
        bbox = dataset_utils.bbox_torchTocv2(bbox).tolist()

    query = query.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    query_size = config.dataset.query_size
    to_tensor = transforms.ToTensor()

    # 混合精度处理
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        query = to_tensor(query).to(device, non_blocking=True)  # [C,H,W]

        if config.dataset.query_padding:
            _, h, w = query.shape
            max_size = max(h, w)
            pad_h = (max_size - h) // 2
            pad_w = (max_size - w) // 2
            query = F.pad(query, (pad_w, pad_w, pad_h, pad_h))
        query = F.interpolate(
            query.unsqueeze(0),
            size=(query_size, query_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    return query.contiguous()


def load_clip(config, clips_img, clips_pcd, clips_pcd_bbox, frame_idx):
    """
    Load frames -> [N,3,H,W] (float, GPU)
    """

    clips = clips_img[frame_idx]
    clips_p = clips_pcd[frame_idx]
    clips_bbox = clips_pcd_bbox[frame_idx]

    clips_origin = clips.clone()

    return clips_origin, clips, clips_p, clips_bbox


def process_inputs(clips, query):
    """
    clips: [B,T,C,H,W]
    query: [C,H,W]
    """
    b, t, c, h, w = clips.shape
    normalization = kornia.enhance.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD).to(device)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        clips = rearrange(clips, 'b t c h w -> (b t) c h w')
        clips = normalization(clips)
        clips = rearrange(clips, '(b t) c h w -> b t c h w', b=b, t=t)

        queries = normalization(query.unsqueeze(0).repeat(b, 1, 1, 1))
    return clips, queries
