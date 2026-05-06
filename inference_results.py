import os, warnings, torch
import torch.nn as nn
warnings.filterwarnings("ignore", message=r"xFormers is available.*", category=UserWarning)

# ==================================================
# 4090 兼容性 & 性能稳定设置
# ==================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 可按需修改 GPU 编号
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "0"  # 禁用 device-side assert
os.environ["NCCL_P2P_DISABLE"] = "1"  # 某些多GPU集群上更稳定
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# 推荐设置：完全绕过 cuDNN Heuristics
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  # 4090支持TF32

# ==================================================
# 其他导入
# ==================================================
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import argparse
import json
import tqdm
from queue import Empty as QueueEmpty

import torch.utils.data
import torch.utils.data.distributed
from torch import multiprocessing as mp
import torch.cuda.amp as amp

from config.config import config, update_config
from utils import exp_utils
from evaluation import eval_utils
from evaluation.task_inference_results import Task
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm_ddp import ClipMatcher

# ==================================================
# ConvTranspose2d 替换（非常重要，防止 4090 报错）
# ==================================================
def _replace_deconv_in_module(module: nn.Module):
    """
    遍历模型，把 ConvTranspose2d 统一替换为 Upsample(scale=2)+Conv2d。
    避开 4090 + cuDNN FIND 报错。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            stride = child.stride if isinstance(child.stride, tuple) else (child.stride, child.stride)
            scale = stride[0]
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

class WorkerWithDevice(mp.Process):
    def __init__(self, config, task_queue, results_queue, worker_id, device_id):
        self.config = config
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))

    def work(self, task_queue, results_queue):

        device = torch.device(f"cuda:{self.device_id}")

        while True:
            try:
                task = task_queue.get(timeout=1)
            except QueueEmpty:  # 专门处理队列空异常
                # 检查是否所有任务已完成
                if task_queue.empty():
                    print(f"Worker {self.worker_id}: task queue empty, exiting")
                    break
                else:
                    # 队列非空但暂时无任务，继续等待
                    continue
            with amp.autocast(dtype=torch.float16):
                key_name = task.run(self.config, device)
            results_queue.put(key_name)
            del task


def get_results(annotations, config):
    total_gpus = torch.cuda.device_count()
    if total_gpus == 0:
        raise RuntimeError("No CUDA device detected! Please ensure RTX 4090 is visible.")
    
    # 自动分配 GPU 列表
    num_gpu_list = list(range(total_gpus))
    print(f"Detected {total_gpus} GPUs -> Using devices {num_gpu_list}")
    num_gpus = len(num_gpu_list)

    mp.set_start_method('spawn', force=True)
    task_queue = mp.Queue()
    for _, annots in annotations.items():
        task = Task(config, annots)
        task_queue.put(task)

    results_queue = mp.Queue()
    num_processes = num_gpus

    pbar = tqdm.tqdm(
        desc=f"Get RT results",
        position=0,
        total=len(annotations),
    )

    workers = [
        WorkerWithDevice(config, task_queue, results_queue, i, num_gpu_list[i % len(num_gpu_list)])
        for i in range(num_processes)
    ]
    # Start workers
    for worker in workers:
        worker.start()
    # Update progress bar
    predicted_rts = {}
    n_completed = 0
    while n_completed < len(annotations):
        try:
            pred = results_queue.get(timeout=5.0)
        except QueueEmpty:
            any_alive = any(w.is_alive() for w in workers)
            if not any_alive:
                exitcodes = [w.exitcode for w in workers]
                raise RuntimeError(f"No more results and no worker alive; worker exitcodes={exitcodes}")
            continue
        else:
            # pred = results_queue.get()
            predicted_rts.update(pred)
            n_completed += 1
            pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    pbar.close()
    return predicted_rts


def format_predictions(annotations, predicted_rts):
    # Format predictions
    predictions = {
        "version": annotations["version"],
        "challenge": "ego4d_vq2d_challenge",
        "results": {"videos": []},
    }
    for v in annotations["videos"]:
        video_predictions = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                auid = a["annotation_uid"]
                apred = {
                    "query_sets": {},
                    "annotation_uid": auid,
                }
                for qid in a["query_sets"].keys():
                    if (auid, qid) in predicted_rts:
                        rt_pred = predicted_rts[(auid, qid)][0].to_json()
                        apred["query_sets"][qid] = rt_pred
                    else:
                        apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}
                clip_predictions["predictions"].append(apred)
            video_predictions["clips"].append(clip_predictions)
        predictions["results"]["videos"].append(video_predictions)
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument('--cfg', default='config/eval.yaml', type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true",help="evaluate model")
    parser.add_argument(
        "--debug", dest="debug", action="store_true",help="evaluate model")
    parser.add_argument(
        "--gt-fg", dest="gt_fg", action="store_true",help="evaluate model")
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    mode = 'eval' if args.eval else 'val'

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    mode = 'test_unannotated' if args.eval else 'val'
    # annotation_path = os.path.join('/vision/hwjiang/episodic-memory/VQ2D/data', 'vq_{}.json'.format(mode))
    annotation_path = '/data_0/pl/VQL_Data/VQL_Data_test' # test
    # annotation_path = '/data_0/pl/VQL_Data/Train'  # train

    # with open(annotation_path) as fp:
    #     annotations = json.load(fp)
    annotations = eval_utils.load_annotations(annotation_path)
    clipwise_annotations_list = eval_utils.convert_annotations_to_clipwise_list(annotations)

    if args.debug:
        clips_list = list(clipwise_annotations_list.keys())
        clips_list = sorted([c for c in clips_list if c is not None])
        clips_list = clips_list[: 20]
        clipwise_annotations_list = {
            k: clipwise_annotations_list[k] for k in clips_list
        }

    predictions_rt = get_results(clipwise_annotations_list, config)
    predictions = format_predictions(annotations, predictions_rt)
    if not args.debug:
        # self.config.output_dir,self.config.dataset.name,"infer_outputs/like_ego4d"
        # test
        with open(os.path.join(config.output_dir, config.dataset.name, "infer_outputs/like_ego4d", '_results.json.gz'), 'w') as fp:
            json.dump(predictions, fp)


