import os
import os.path as osp
import json
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

def load_annotations(data_path) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    batch_ids = _get_available_batch_ids(data_path)
    batch_ids = [1]
    for batch_id in batch_ids:
        scene_ids = _get_available_scene_ids(data_path, batch_id)
        for scene_id in tqdm(scene_ids, desc='[%6s]Loading annos' % f'batch{batch_id}'):
            search_file_dir = osp.join(
                data_path, f"batch{batch_id}", 'lable', 'o_s',
                'Seq_%06d' % scene_id, 'Seq_%06d.json' % scene_id
            )
            template_file_dir = osp.join(
                data_path, f"batch{batch_id}", 'lable', 'o_s_t',
                'Seq_%06d' % scene_id, 'Seq_%06d.json' % scene_id
            )
            search_data = pd.read_json(search_file_dir)
            template_data = pd.read_json(template_file_dir)
            search_objects_series = search_data['objects']
            template_objects_series = template_data['objects']

            search_objects = []
            template_objects = []
            for obj_list, i in zip(search_objects_series, range(len(search_objects_series))):
                if len(obj_list) > 0 and isinstance(obj_list, list):
                    obj_list[0]['scene'] = scene_id
                    obj_list[0]['frame'] = i + 1
                    obj_list[0]['batch_id'] = batch_id
                    search_objects.extend(obj_list)
                else:
                    none_objects = {'id': None, 'type': None, 'classId': None, 'className': None, 'trackId': None, 'trackName': None, 'classValues': None, 'contour': None, 'modelConfidence': None, 'modelClass': '', 'scene': scene_id, 'frame': i + 1, 'batch_id': batch_id}
                    search_objects.append(none_objects)

            for obj_list, i in zip(template_objects_series, range(len(template_objects_series))):
                if len(obj_list) > 0 and isinstance(obj_list, list):
                    obj_list[0]['scene'] = scene_id
                    obj_list[0]['frame'] = i + 1
                    obj_list[0]['batch_id'] = batch_id
                    template_objects.extend(obj_list)

            if len(template_objects) > 1:
                pcd_folder = _get_pcd_folder_path(data_path, batch_id, scene_id, 'o_s_t')
                tem_file_dir = osp.join(
                    data_path, f"batch{batch_id}", 'img', 'o_s_t', 'Seq_%06d' % scene_id, pcd_folder)
                file_names = os.listdir(tem_file_dir)[0]
                frame_id = int(file_names.split('.')[0])
                template_objects = [item for item in template_objects if item.get("frame") == frame_id]

            if not search_objects or len(search_objects) == 0:
                continue
            if not template_objects or len(template_objects) == 0:
                continue

            search_df = pd.DataFrame(search_objects)
            template_df = pd.DataFrame(template_objects)
            search_tracklet_anno = [frame_anno.to_dict() for _, frame_anno in search_df.iterrows()]
            template_tracklet_anno = [frame_anno.to_dict() for _, frame_anno in template_df.iterrows()]
            """
            {
                "videos": [
                    {
                    "video_uid": "video_001",
                    "clips": [
                        {
                        "clip_uid": "clip_001",
                        "video_start_sec": 0.0,
                        "video_end_sec": 10.0,
                        "clip_fps": 30,
                        "annotations": [
                            {
                            "annotation_uid": "anno_001",
                            "query_sets": {
                                "Q0": {
                                "is_valid": true,
                                "query_frame": 15,
                                "visual_crop": {
                                    "x": 100,
                                    "y": 200,
                                    "width": 50,
                                    "height": 50
                                },
                                "response_track": {
                                    "frames": [15, 16, 17],
                                    "bboxes": [
                                    {"x": 100, "y": 200, "width": 50, "height": 50},
                                    {"x": 105, "y": 205, "width": 50, "height": 50},
                                    {"x": 110, "y": 210, "width": 50, "height": 50}
                                    ]
                                }
                                },
                                "Q1": {
                                "is_valid": false, // 无效查询，会被跳过
                                "query_frame": 20,
                                "visual_crop": { ... }
                                }
                            }
                            }
                        ]
                        },
                        {
                        "clip_uid": "clip_002",
                        "video_start_sec": 10.0,
                        "video_end_sec": 20.0,
                        "clip_fps": 30,
                        "annotations": [ ... ]
                        }
                    ]
                    },
                    {
                    "video_uid": "video_002",
                    "clips": [ ... ]
                    }
                ]
            }
            """
            # annotations.append({'batch_id': batch_id, 'scene_id': scene_id, 'search': search_tracklet_anno, 'template': template_tracklet_anno[0]})
            # 一个Seq作为一个视频片段video_uid，一共3个有效轨迹段，根据是否有标注信息判定，划分为3个clip_uid
            # 假设一共300帧，有效帧为10-80 90-120 150-180，则划分为 1-90 91-149 150-300
            video_uid = f"batch{batch_id}_scene{scene_id}"
            clip_start_idx = [1]; clip_start_valid_idx = []
            clip_end_idx = []; clip_end_valid_idx = []    
            last_valid = False; valid_start = False
            for idx, frame_anno in enumerate(search_tracklet_anno):
                if frame_anno.get('contour') is not None and last_valid is False:
                    clip_start_valid_idx.append(idx+1)
                    if not valid_start:
                        valid_start = True
                        last_valid = True
                        continue
                    clip_start_idx.append(idx+1)
                    last_valid = True
                elif frame_anno.get('contour') is None and last_valid is True:
                    clip_end_valid_idx.append(idx)
                    last_valid = False
            for idx in clip_start_idx[1:]:
                clip_end_idx.append(idx - 1)
            clip_end_idx.append(len(search_tracklet_anno))
            if valid_start and len(clip_start_valid_idx) > len(clip_end_valid_idx):
                clip_end_valid_idx.append(len(search_tracklet_anno))
            assert len(clip_start_valid_idx) == len(clip_end_valid_idx), f"Error in clip split: {clip_start_idx}, {clip_end_idx}"
            clips = []
            for clip_idx in range(len(clip_start_valid_idx)):
                clip_uid = f"{video_uid}_clip{clip_idx+1}"
                start_idx = clip_start_idx[clip_idx]
                end_idx = clip_end_idx[clip_idx]
                clip_fps = None
                video_start_frame = start_idx
                video_end_frame = end_idx + 1
                # clip_search_annos = search_tracklet_anno[start_idx:end_idx+1]
                annots = []
                annotation_uid = f"{clip_uid}_anno1"
                response_track = []
                for idx in range(clip_start_valid_idx[clip_idx], clip_end_valid_idx[clip_idx]+1):
                    frame_anno = search_tracklet_anno[idx - 1]
                    assert frame_anno.get('contour') is not None, f"Error frame without contour in response track: clip {clip_uid}, frame {idx}"
                    x, y, z = frame_anno['contour']['center3D']['x'], frame_anno['contour']['center3D']['y'], frame_anno['contour']['center3D']['z']
                    w, l, h = frame_anno['contour']['size3D']['x'], frame_anno['contour']['size3D']['y'], frame_anno['contour']['size3D']['z']
                    roll, pitch, yaw = frame_anno['contour']['rotation3D']['x'], frame_anno['contour']['rotation3D']['y'], frame_anno['contour']['rotation3D']['z']
                    response_track.append({'frame_number' : idx,'x': x, 'y': y, 'z': z, 'w': w, 'l': l, 'h': h, 'roll': roll, 'pitch': pitch, 'yaw': yaw})
                query_sets = {
                    "Q0": {
                        "is_valid": True,
                        "query_frame": video_end_frame + 1,
                        "visual_crop": {
                            "frame_number": video_end_frame + 1,
                            "x" : template_tracklet_anno[0]['contour']['center3D']['x'],
                            "y" : template_tracklet_anno[0]['contour']['center3D']['y'],
                            "z" : template_tracklet_anno[0]['contour']['center3D']['z'],
                            "w" : template_tracklet_anno[0]['contour']['size3D']['x'],
                            "l" : template_tracklet_anno[0]['contour']['size3D']['y'],
                            "h" : template_tracklet_anno[0]['contour']['size3D']['z'],
                            "roll": template_tracklet_anno[0]['contour']['rotation3D']['x'],
                            "pitch": template_tracklet_anno[0]['contour']['rotation3D']['y'],
                            "yaw": template_tracklet_anno[0]['contour']['rotation3D']['z']
                        },
                        "response_track": response_track
                    }
                }
                annots.append({
                    "annotation_uid": annotation_uid,
                    "query_sets": query_sets
                })
                clip_dict = {
                    "clip_uid": clip_uid,
                    "video_start_sec": video_start_frame,
                    "video_end_sec": video_end_frame,
                    "clip_fps": clip_fps,
                    "annotations": annots
                }
                clips.append(clip_dict)
            video_dict = {
                "video_uid": video_uid,
                "clips": clips
            }
            annotations.append(video_dict)
        return_dict = {
            "version": "1.0",
            "videos": annotations
        }
    return return_dict

def _get_pcd_folder_path(data_path, batch_id: int, scene_id: int, data_type: str = 'o_s') -> str:
    """
    自动选择合适的pcd文件夹路径，优先 lidar_point_cloud_0，其次 point_cloud_bin。
    """
    base_path = osp.join(data_path, f"batch{batch_id}", 'img', data_type, 'Seq_%06d' % scene_id)
    lidar_folder = osp.join(base_path, 'lidar_point_cloud_0')
    point_cloud_bin_folder = osp.join(base_path, 'point_cloud_bin')

    if osp.exists(lidar_folder):
        files = os.listdir(lidar_folder)
        pcd_files = [f for f in files if f.endswith('.pcd')]
        if len(pcd_files) > 0:
            return 'lidar_point_cloud_0'

    if osp.exists(point_cloud_bin_folder):
        files = os.listdir(point_cloud_bin_folder)
        pcd_files = [f for f in files if f.endswith('.pcd')]
        if len(pcd_files) > 0:
            return 'point_cloud_bin'
    return 'lidar_point_cloud_0'

def _get_available_batch_ids(data_path) -> List[int]:
    """
    扫描数据目录，获取所有可用的批次ID。
    """
    batch_ids: List[int] = []
    search_dir = osp.join(data_path)
    if osp.exists(search_dir):
        for folder_name in os.listdir(search_dir):
            if folder_name.startswith('batch') and osp.isdir(osp.join(search_dir, folder_name)):
                batch_id = int(folder_name[5:])
                batch_ids.append(batch_id)
    batch_ids.sort()
    return batch_ids

def _get_available_scene_ids(data_path, batch_id: int) -> List[int]:
    """
    扫描数据目录，获取所有可用的场景ID。
    """
    scene_ids: List[int] = []
    search_dir = osp.join(data_path, f"batch{batch_id}", 'lable', 'o_s')
    if osp.exists(search_dir):
        for folder_name in os.listdir(search_dir):
            if folder_name.startswith('Seq_') and osp.isdir(osp.join(search_dir, folder_name)):
                scene_id = int(folder_name.split('_')[1])
                scene_ids.append(scene_id)
    scene_ids.sort()
    # scene_ids = [1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998,1998, 1998, 1998]  # DEBUG
    return scene_ids

def convert_annotations_to_clipwise_list(annotations):
    clipwise_annotations_list = {}
    for v in annotations["videos"]:
        vuid = v["video_uid"]
        for c in v["clips"]:
            cuid = c["clip_uid"]
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    if not q["is_valid"]:
                        continue
                    curr_q = {
                        "metadata": {
                            "video_uid": vuid,
                            "video_start_sec": c["video_start_sec"],
                            "video_end_sec": c["video_end_sec"],
                            "clip_fps": c["clip_fps"],
                            "query_set": qid,
                            "annotation_uid": aid,
                        },
                        "clip_uid": cuid,
                        "query_frame": q["query_frame"],
                        "visual_crop": q["visual_crop"],
                    }
                    if "response_track" in q:
                        curr_q["response_track"] = q["response_track"]
                    if cuid not in clipwise_annotations_list:
                        clipwise_annotations_list[cuid] = []
                    clipwise_annotations_list[cuid].append(curr_q)
    return clipwise_annotations_list

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
