#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集中的 GT 注释转换为“预测文件”同款格式（results.videos...），
用于在 evaluate.py 中进行 "gt vs gt" 的自测评估。

输出结构满足 evaluate.validate_model_predictions 的要求：
{
  "version": "1.0",
  "challenge": "ego4d_vq2d_challenge",
  "results": {
	"videos": [
	  { "video_uid": ..., "clips": [
		  { "clip_uid": ..., "predictions": [
			  {
				"annotation_uid": ...,
				"query_sets": {
				  "Q0": { "bboxes": [...], "score": 1.0 },
				  ...
				}
			  }, ...
		  ]}
	  ]}
	]
  }
}

使用方式：
  python build_gt_for_pred.py \
	--data-path /data_0/pl/VQL_Data/VQL_Data_test \
	--out output/VQLOC/infer_outputs/like_ego4d/_results.json.gz

说明：
- 这里与 inference_results.py 保持一致，尽管扩展名为 .json.gz，我们仍写入普通 JSON。
- 评分 score 固定为 1.0；bboxes 来源于 GT 的 response_track（3D box，字段映射为 fno/x/y/z/w/l/h/roll/pitch/yaw）。
"""

import os
import os.path as osp
import json
import argparse
from typing import Dict, Any

from .evaluation import eval_utils
from .evaluation.structures import BBox3D, ResponseTrack


def parse_args():
	parser = argparse.ArgumentParser(description="Build prediction-style JSON from GT annotations")
	parser.add_argument(
		"--data-path",
		type=str,
		default="/data_0/pl/VQL_Data/VQL_Data_test",
		help="Root directory of the dataset (contains batch*/ folders)",
	)
	parser.add_argument(
		"--out",
		type=str,
		default="output/VQLOC/infer_outputs/like_ego4d/_gtresults.json.gz",
		help="Path to write the generated prediction-style JSON (dirs will be created)",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Pretty-print JSON (larger file, useful for inspection)",
	)
	return parser.parse_args()


def gt_to_pred_style(annotations: Dict[str, Any]) -> Dict[str, Any]:
	"""将 annotations（load_annotations 输出的 GT 结构）转换为预测风格结构。

	关键点：
	- 仅对 is_valid 的 query_set 输出内容；
	- bboxes 来自 q["response_track"], 字段名 frame_number -> fno；
	- score 统一设为 1.0；
	- 组织成 evaluate.py 期望的 results.videos.clips.predictions 列表。
	"""

	predictions = {
		"version": annotations["version"],
		"challenge": "ego4d_vq2d_challenge",
		"results": {"videos": []},
	}

	for v in annotations.get("videos", []):
		video_predictions = {"video_uid": v["video_uid"], "clips": []}
		for c in v.get("clips", []):
			clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
			for a in c.get("annotations", []):
				auid = a["annotation_uid"]
				apred = {"query_sets": {}, "annotation_uid": auid}
				for qid, q in a.get("query_sets", {}).items():
					if not q.get("is_valid", False):
						# 非有效查询不参与评价，跳过
						continue

					# 将 GT 的 response_track -> ResponseTrack JSON（含 bboxes, score）
					bboxes = []
					for rf in q.get("response_track", []):
						# GT 字段名：frame_number -> 预测 JSON 里使用 BBox3D.to_json 的 fno
						bbox = BBox3D(
							rf["frame_number"],
							rf["x"],
							rf["y"],
							rf["z"],
							rf["w"],
							rf["l"],
							rf["h"],
							rf["roll"],
							rf["pitch"],
							rf["yaw"],
						)
						bboxes.append(bbox)

					# 保证轨迹是按帧连续的，ResponseTrack 会进行检查
					rt = ResponseTrack(bboxes, score=1.0)
					apred["query_sets"][qid] = rt.to_json()

				clip_predictions["predictions"].append(apred)
			video_predictions["clips"].append(clip_predictions)
		predictions["results"]["videos"].append(video_predictions)

	return predictions


def main():
	args = parse_args()

	# 1) 加载 GT 注释（使用与推理一致的读取逻辑）
	print(f"[build_gt_for_pred] Loading annotations from: {args.data_path}")
	annotations = eval_utils.load_annotations(args.data_path)

	# 2) 转换为预测风格 JSON
	print("[build_gt_for_pred] Converting GT -> prediction-style JSON ...")
	predictions = gt_to_pred_style(annotations)

	# 3) 写文件（普通 JSON，路径中间目录会自动创建）
	out_dir = osp.dirname(args.out)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	print(f"[build_gt_for_pred] Writing predictions to: {args.out}")
	with open(args.out, "w", encoding="utf-8") as f:
		if args.pretty:
			json.dump(predictions, f, ensure_ascii=False, indent=2)
		else:
			json.dump(predictions, f)

	# 4) 简要统计
	n_videos = len(predictions.get("results", {}).get("videos", []))
	n_clips = sum(len(v.get("clips", [])) for v in predictions["results"]["videos"])
	print(f"[build_gt_for_pred] Done. videos={n_videos}, clips={n_clips}")


if __name__ == "__main__":
	main()

