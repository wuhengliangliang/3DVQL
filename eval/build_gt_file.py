#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build GT file for VQLoC/VQ2D-style evaluation from the original dataset labels.

This script scans your dataset directory (original label layout) and emits a
single JSON containing:
{
  "version": "1.0",
  "videos": [
    { "video_uid": ..., "clips": [ { "clip_uid": ..., "annotations": [...] }, ... ] },
    ...
  ]
}

The per-clip annotation structure matches what the inference pipeline expects
(via evaluation.eval_utils.load_annotations). In particular, it preserves the
3D-oriented fields that your pipeline uses (visual_crop and response_track with
3D center/size/rotation), so downstream inference/result-formatting code works
out-of-the-box.

Usage:
  python build_gt_file.py \
    --data-path /data_0/pl/VQL_Data/VQL_Data_test \
    --out output/VQLOC/infer_outputs/like_ego4d/_gt.json.gz

Notes:
- The output extension ".json.gz" is kept for symmetry with predictions, but
  we write plain JSON (no compression), same as inference_results.py.
- If you later need a 2D VQ2D-style GT (x,y,width,height with frame_number),
  we can extend this script to project 3D boxes using calibration files.
"""
import os
import os.path as osp
import json
import argparse

from evaluation import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Build GT file from raw labels")
    parser.add_argument(
        "--data-path",
        type=str,
        default='/mnt/data_2/pl/VQL_Data_test/VQL_Data_test',
        help="Root directory of the dataset (contains batch*/ folders)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/VQLOC/infer_outputs/like_ego4d/_gt.json.gz",
        help="Path to write the generated GT JSON (dirs will be created)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (larger file, useful for inspection)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load annotations using the same logic as inference
    print(f"[build_gt_file] Scanning annotations from: {args.data_path}")
    annotations = eval_utils.load_annotations(args.data_path)

    # 2) Ensure output directory exists
    out_dir = osp.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 3) Save JSON (plain, no compression like inference_results.py)
    print(f"[build_gt_file] Writing GT to: {args.out}")
    if args.pretty:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(annotations, f)

    # 4) Brief stats
    n_videos = len(annotations.get("videos", []))
    n_clips = sum(len(v.get("clips", [])) for v in annotations.get("videos", []))
    print(f"[build_gt_file] Done. videos={n_videos}, clips={n_clips}")


if __name__ == "__main__":
    main()
