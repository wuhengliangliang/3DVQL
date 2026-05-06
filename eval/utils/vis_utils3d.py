import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
from dataset import dataset_utils
import torch
from einops import rearrange
import numpy as np
import math
from utils.mathConvert import (
    compute_3d_box_vertices,
    compute_rotation_matrix_from_directions,
    project_3d_to_2d,
)
intrinsic = np.array([[1044.61, 0, 642.33],
              [0, 1046.04, 367.719],
              [0, 0, 1]])
extrinsic = np.array([[-0.019445, -0.999799, -0.00484243, 0.0308751],
              [-0.0130265, 0.00509626, -0.999903, -0.326309],
              [0.999617, -0.0263807, -0.0130804, -0.017514],
              [0, 0, 0, 1]])
# def draw_3d_bbox_on_img(rgb, bbox):
#     """
#     rgb: [H,W,3], numpy array
#     bbox: [x, y, z, l, w, h, roll, pitch, yaw]
#     return: rgb with 2D bbox drawn
#     """
#     rgb = rgb.copy()
#     H, W, _ = rgb.shape
#     bbox2d = compute_rgb_bbox(bbox)  # [x_min, y_min, x_max, y_max]
#     bbox2d = (bbox2d * np.array([W, H, W, H])).astype(np.int32)
#     x_min, y_min, x_max, y_max = bbox2d
#     x_min = int(np.clip(x_min, 0, W-1))
#     x_max = int(np.clip(x_max, 0, W-1))
#     y_min = int(np.clip(y_min, 0, H-1))
#     y_max = int(np.clip(y_max, 0, H-1))
#     cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     return rgb
# def renormize_bbox(bbox):
#     bbox = bbox.clone()
#     # bbox[..., 0] = bbox[..., 0] * (space_range[1][0] - space_range[0][0]) + space_range[0][0]
#     # bbox[..., 1] = bbox[..., 1] * (space_range[1][1] - space_range[0][1]) + space_range[0][1]
#     # bbox[..., 2] = bbox[..., 2] * (space_range[1][2] - space_range[0][2]) + space_range[0][2]
#     bbox[..., 6] = bbox[..., 6] * 2 * math.pi
#     bbox[..., 7] = bbox[..., 7] * 2 * math.pi
#     bbox[..., 8] = bbox[..., 8] * 2 * math.pi
#     return bbox
def compute_3d_box_vertices_withlist(bbox):
    """
    bbox: [x, y, z, l, w, h, roll, pitch, yaw]
    return: [8, 3]
    """
    # bbox = renormize_bbox(torch.tensor(bbox)).numpy()
    cen = bbox[0:3]
    siz = bbox[3:6]
    rot = bbox[6:9]
    vertices = compute_3d_box_vertices(cen, siz, rot)  # [8,3]
    return vertices

def standardize_bbox(bbox, H, W):
    """
    bbox: [8, 2] origin axis
    return: [8, 2] resize, pad axis
    """
    origin_w = 1280
    origin_h = 720
    target_w = 448
    target_h = 448
    bbox = bbox.clone()
    max_size, min_size = max(origin_h, origin_w), min(origin_h, origin_w)
    pad_height = True if origin_h < origin_w else False
    pad_size = (max_size - min_size) // 2
    if pad_height:
        pad_input = [0, pad_size] * 2
        bbox[:, 1] += pad_size
    else:
        pad_input = [pad_size, 0] * 2
        bbox[:, 0] += pad_size
    bbox = bbox / max_size
    bbox[:, 0] = bbox[:, 0] * target_w
    bbox[:, 1] = bbox[:, 1] * target_h
    # clip
    bbox[:, 0] = torch.clamp(bbox[:, 0], 0, target_w - 1)
    bbox[:, 1] = torch.clamp(bbox[:, 1], 0, target_h - 1)
    return bbox

def vis_pred_clip(sample, preds, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    clip = sample['clip_origin'].detach().cpu()        # [B,T,3,H,W]
    query = sample['query_origin'].detach().cpu()      # [B,3,H2,W2]
    query_aug = sample['query'].detach().cpu()         # [B,3,H2,W2]
    bbox = sample['clip_pcd_bbox'].detach().cpu()          # [B,T,9]
    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    B, T, _, H, W = clip.shape
    _, _, H2, W2 = query_aug.shape
    
    for pred_id, pred in enumerate(preds):

        bbox_pred = pred['bbox'].detach().cpu()            # [B,T,4]
        prob_pred = pred['prob'].detach().cpu()            # [B,T]

        for i in range(B):
            frames = []
            cur_clip, cur_query = clip[i], query[i]                                     # [T,3,H,W], [3,H2,W2]
            cur_bbox, cur_bbox_pred = bbox[i], bbox_pred[i]                             # [T,9] [x, y, z, l, w, h, roll, pitch, yaw]
            cur_prob, cur_prob_pred = prob[i], prob_pred[i]                             # [T]

            cur_query = cur_query.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()        # [H2,W2,3]
            for j in range(T):
                # draw clips with bbox
                img = cur_clip[j].clamp(min=0.0, max=1.0)                               
                img = img.permute(1,2,0).numpy()                # [H,W,3]
                fig, ax = plt.subplots(1,2, dpi=100)
                fig.suptitle('Prob: gt {:.3f}, pred {:.3f}'.format(cur_prob[j].item(), torch.sigmoid(cur_prob_pred[j]).item()), fontsize=20)
                ax[0].imshow(img)
                ax[1].imshow(cur_query)
                if cur_prob[j].item() > 0.5:
                    # draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[j], H, W)  # [4]
                    draw_bbox_corner_gt = compute_3d_box_vertices_withlist(cur_bbox[j].numpy())  # [8, 3]
                    draw_bbox_corner2d_gt = project_3d_to_2d(draw_bbox_corner_gt, intrinsic, extrinsic)  # [8, 2]
                    draw_bbox_corner2d_gt = standardize_bbox(draw_bbox_corner2d_gt, H, W).numpy()
                    # 0: front-top-right
                    # 1: front-top-left
                    # 2: back-top-left
                    # 3: back-top-right
                    # 4: front-bottom-right
                    # 5: front-bottom-left
                    # 6: back-bottom-left
                    # 7: back-bottom-right  
                    # 16条线，3D box
                    edges = [(0,1),(1,2),(2,3),(3,0),
                            (4,5),(5,6),(6,7),(7,4),
                            (0,4),(1,5),(2,6),(3,7)]
                    for edge in edges:
                        ax[0].plot(*zip(draw_bbox_corner2d_gt[edge[1]], draw_bbox_corner2d_gt[edge[0]]), color='r')
                    # rect = patches.Rectangle((draw_bbox_gt[1], draw_bbox_gt[0]), 
                    #                         draw_bbox_gt[3]-draw_bbox_gt[1], draw_bbox_gt[2]-draw_bbox_gt[0], 
                    #                         linewidth=1, edgecolor='r', facecolor='none')
                    # ax[0].add_patch(rect)
                if cur_prob[j].item() > 0.5:
                    # draw_bbox_pred = dataset_utils.recover_bbox(cur_bbox_pred[j], H, W)  # [4]
                    # rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                    #                         draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                    #                         linewidth=1, edgecolor='g', facecolor='none')
                    draw_bbox_corner_pred = compute_3d_box_vertices_withlist(cur_bbox_pred[j].numpy())  # [8, 3]
                    draw_bbox_corner2d_pred = project_3d_to_2d(draw_bbox_corner_pred, intrinsic, extrinsic)  # [8, 2]
                    draw_bbox_corner2d_pred = standardize_bbox(draw_bbox_corner2d_pred, H, W).numpy()
                    # rect = patches.Polygon(draw_bbox_corner2d_pred, closed=True, linewidth=1, edgecolor='g', facecolor='none')
                    # ax[0].add_patch(rect)
                    edges = [(0,1),(1,2),(2,3),(3,0),
                            (4,5),(5,6),(6,7),(7,4),
                            (0,4),(1,5),(2,6),(3,7)]
                    for edge in edges:
                        ax[0].plot(*zip(draw_bbox_corner2d_pred[edge[0]], draw_bbox_corner2d_pred[edge[1]]), color='g')
                if torch.sigmoid(cur_prob_pred[j]).item() > 0.5:
                    # draw_bbox_pred = dataset_utils.recover_bbox(cur_bbox_pred[j], H, W)  # [4]
                    # rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                    #                         draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                    #                         linewidth=1, edgecolor='b', facecolor='none')
                    draw_bbox_corner_pred = compute_3d_box_vertices_withlist(cur_bbox_pred[j].numpy())  # [8, 3]
                    draw_bbox_corner2d_pred = project_3d_to_2d(draw_bbox_corner_pred, intrinsic, extrinsic)  # [8, 2]
                    draw_bbox_corner2d_pred = standardize_bbox(draw_bbox_corner2d_pred, H, W).numpy()
                    # rect = patches.Polygon(draw_bbox_corner2d_pred, closed=True, linewidth=1, edgecolor='b', facecolor='none')
                    # ax[0].add_patch(rect)
                    edges = [(0,1),(1,2),(2,3),(3,0),
                            (4,5),(5,6),(6,7),(7,4),
                            (0,4),(1,5),(2,6),(3,7)]
                    for edge in edges:
                        ax[0].plot(*zip(draw_bbox_corner2d_pred[edge[0]], draw_bbox_corner2d_pred[edge[1]]), color='b')
                # plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'tmp.png'))
                plt.close()
                frames.append(cv2.imread(os.path.join(output_dir, 'tmp.png'))[...,::-1])
            save_name = os.path.join(output_dir, '{}_{}_{}.gif'.format(iter_num, i, pred_id))
            imageio.mimsave(save_name, frames, 'GIF', duration=0.2)

def vis_pred_scores(sample, preds, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    B, T = prob.shape

    for pred_id, pred in enumerate(preds):
        prob_pred = pred['prob'].detach().cpu()            # [B,T]
        if 'gt_iou' in pred.keys():
            prob_iou = pred['gt_iou'].detach().cpu()            # [B,T]
        if 'prob_refine' in pred.keys():
            prob_refine = pred['prob_refine'].detach().cpu()            # [B,T]

        for i in range(B):
            cur_prob, cur_prob_pred = prob[i].numpy(), torch.sigmoid(prob_pred[i]).numpy()     # [T]
            x = np.arange(T)
            plt.plot(x, cur_prob_pred, marker=None, color='b', label='pred')
            plt.plot(x, cur_prob, marker=None, color='r', label='gt')
            if 'prob_refine' in pred.keys():
                cur_prob_refine = torch.sigmoid(prob_refine[i]).numpy()
                plt.plot(x, cur_prob_refine, marker=None, color='g', label='pred')
            if 'gt_iou' in pred.keys():
                cur_prob_iou = prob_iou[i].numpy() * 0.9
                plt.plot(x, cur_prob_iou, marker=None, color='c', label='pred')
            plt.xlabel('number of frames')
            plt.ylabel('occurance score')
            plt.ylim((0.0, 1.05))
            plt.legend(loc='best')
            save_name = os.path.join(output_dir, '{}_{}_{}.jpg'.format(iter_num, i, pred_id))
            plt.savefig(save_name)
            plt.close()

def vis_pred_clip_inference(clips, queries, pred, save_path, iter_num):
    #clips = clips.detach().cpu()            # [b,t,c,h,w]
    queries = queries.detach().cpu()        # [c,h,w]
    # bbox = pred['bbox_raw']                 # [b*t,9]
    # prob = torch.sigmoid(pred['prob_raw'])  # [b*t]
    bbox = pred['bbox']                 # [b*t,9]
    prob = torch.sigmoid(pred['prob'])  # [b*t]
    save_name = save_path + f'_{iter_num}.mp4'
    writer = imageio.get_writer(save_name, fps=5)

    #clips = rearrange(clips, 'b t c h w -> (b t) c h w')

    T, _, H, W = clips.shape
    _, H2, W2 = queries.shape

    frames = []
    for i in range(T):
        cur_clip = clips[i].clamp(min=0.0, max=1.0).permute(1,2,0).numpy()
        cur_query = queries.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()
        cur_bbox = bbox[i]#.clamp(min=0.0, max=1.0)
        cur_prob = prob[i]

        fig, ax = plt.subplots(1,2)
        fig.suptitle('Prob {:.3f}'.format(cur_prob.item()), fontsize=20)
        ax[0].imshow(cur_clip)
        ax[1].imshow(cur_query)
        if cur_prob.item() > 0.5:
            draw_bbox_pred = cur_bbox #dataset_utils.recover_bbox(cur_bbox, H, W)  # [9]
            draw_bbox_pred = compute_3d_box_vertices_withlist(draw_bbox_pred.numpy())  # [8, 3]
            draw_bbox_pred = project_3d_to_2d(draw_bbox_pred, intrinsic, extrinsic)  # [8, 2]
            draw_bbox_pred = standardize_bbox(draw_bbox_pred, H, W).numpy()
            # rect = patches.Rectangle((draw_bbox_pred[0], draw_bbox_pred[1]), 
            #                           draw_bbox_pred[2]-draw_bbox_pred[0], draw_bbox_pred[3]-draw_bbox_pred[1], 
            #                           linewidth=1, edgecolor='b', facecolor='none')
            edges = [(0,1),(1,2),(2,3),(3,0),
                    (4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7)]
            for edge in edges:
                ax[0].plot(*zip(draw_bbox_pred[edge[0]], draw_bbox_pred[edge[1]]), color='b')
            # ax[0].add_patch(rect)
        plt.savefig(save_path + '_tmp.jpg')
        plt.close()
        writer.append_data(cv2.imread(save_path + '_tmp.jpg')[...,::-1])
    writer.close()

def vis_pred_topk(sample, pred, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    topk_dic=pred['topk_dict']
    original_top_k_indices=topk_dic['original_top_k_indices']
    original_top_k_scores=topk_dic['original_top_k_scores']
    original_top_k_bbox=topk_dic['original_top_k_bbox']

    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    clip = sample['clip_origin'].detach().cpu()        # [B,T,3,H,W]
    query = sample['query_origin'].detach().cpu()      # [B,3,H2,W2]
    # query_aug = sample['query'].detach().cpu()         # [B,3,H2,W2]
    bbox = sample['clip_pcd_bbox'].detach().cpu()          # [B,T,9]
    # bbox_pred = pred['bbox'].detach().cpu()            # [B,T,9]
    top_k_indices_cpu = original_top_k_indices.detach().cpu()            # [B,k]
    top_k_scores_cpu = original_top_k_scores.detach().cpu()            # [B,k]
    top_k_bbox_cpu = original_top_k_bbox.detach().cpu()            # [B,k,9]

    


    B, T, _, H, W = clip.shape
    # _, _, H2, W2 = query_aug.shape
    _,K=top_k_indices_cpu.shape

    for i in range(B):
        frames = []
        cur_clip, cur_query = clip[i], query[i]                                     # [T,3,H,W], [3,H2,W2]
        cur_bbox = bbox[i]     # [T,4]
        cur_prob = prob[i]                             # [T]

        cur_topk_scores = top_k_scores_cpu[i] # [k]
        cur_topk_indices = top_k_indices_cpu[i] # [k]
        cur_topk_bbox = top_k_bbox_cpu[i] # [k,9]


        cur_query = cur_query.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()        # [H2,W2,3]
        for j in range(K):
            # draw clips with bbox
            ori_idx=cur_topk_indices[j]
            img = cur_clip[ori_idx].clamp(min=0.0, max=1.0)                               
            img = img.permute(1,2,0).numpy()                # [H,W,3]
            fig, ax = plt.subplots(1,2, dpi=100)
            # draw_bbox_pred = dataset_utils.recover_bbox(cur_topk_bbox[j], H, W)
            # draw_bbox_pred = compute_3d_box_vertices_withlist(cur_topk_bbox[j].numpy())  # [8, 3]
            # draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[ori_idx], H, W)
            # draw_bbox_gt = compute_3d_box_vertices_withlist(cur_bbox[ori_idx].numpy())  # [8, 3]

            # Use .item() to get a single numeric value
            cur_prob_value = cur_prob[ori_idx].item()
            cur_pred_value = torch.sigmoid(cur_topk_scores[j]).item()
            draw_bbox_values = [x.item() for x in cur_topk_bbox[j]]
            draw_bbox_values_gt = [x.item() for x in cur_bbox[ori_idx]]
            

            # Then use these values ​​to format the string
            # title_str = 'Prob: gt {:.3f}:[({:.0f},{:.0f}),({:.0f},{:.0f})]\n pred {:.3f}:[({:.0f},{:.0f}),({:.0f},{:.0f})]'.format(
            #     cur_prob_value,
            #     draw_bbox_values_gt[0],draw_bbox_values_gt[1],
            #     draw_bbox_values_gt[2],draw_bbox_values_gt[3],
            #     cur_pred_value,
            #     draw_bbox_values[0], draw_bbox_values[1],
            #     draw_bbox_values[2], draw_bbox_values[3],
            # )
            title_str = 'Prob: gt {:.3f}:[center({:.1f},{:.1f},{:.1f}),size({:.1f},{:.1f},{:.1f}),rot({:.1f},{:.1f},{:.1f})]\n pred {:.3f}:[center({:.1f},{:.1f},{:.1f}),size({:.1f},{:.1f},{:.1f}),rot({:.1f},{:.1f},{:.1f})]'.format(
                cur_prob_value,
                draw_bbox_values_gt[0],
                draw_bbox_values_gt[1],
                draw_bbox_values_gt[2],
                draw_bbox_values_gt[3],
                draw_bbox_values_gt[4],
                draw_bbox_values_gt[5],
                draw_bbox_values_gt[6],
                draw_bbox_values_gt[7],
                draw_bbox_values_gt[8],
                cur_pred_value,
                draw_bbox_values[0],
                draw_bbox_values[1],
                draw_bbox_values[2],
                draw_bbox_values[3],
                draw_bbox_values[4],
                draw_bbox_values[5],
                draw_bbox_values[6],
                draw_bbox_values[7],
                draw_bbox_values[8],
            )

            # Set the title using the corrected string
            fig.suptitle(title_str, fontsize=10)

            ax[0].imshow(img)
            ax[1].imshow(cur_query)
            if cur_prob[ori_idx].item() > 0.5:
                # draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[ori_idx], H, W)  # [4]
                # rect = patches.Rectangle((draw_bbox_gt[1], draw_bbox_gt[0]), 
                #                          draw_bbox_gt[3]-draw_bbox_gt[1], draw_bbox_gt[2]-draw_bbox_gt[0], 
                #                          linewidth=1, edgecolor='r', facecolor='none')
                draw_bbox_gt = compute_3d_box_vertices_withlist(cur_bbox[ori_idx].numpy())  # [8, 3]
                draw_bbox_gt = project_3d_to_2d(draw_bbox_gt, intrinsic, extrinsic)  # [8, 2]
                draw_bbox_gt = standardize_bbox(draw_bbox_gt, H, W).numpy()
                edges = [(0,1),(1,2),(2,3),(3,0),
                        (4,5),(5,6),(6,7),(7,4),
                        (0,4),(1,5),(2,6),(3,7)]
                for edge in edges:
                    ax[0].plot(*zip(draw_bbox_gt[edge[0]], draw_bbox_gt[edge[1]]), color='r')
                # ax[0].add_patch(rect)
            if cur_prob[ori_idx].item() > 0.5:
                # draw_bbox_pred = dataset_utils.recover_bbox(cur_topk_bbox[j], H, W)  # [4]
                # rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                #                          draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                #                          linewidth=1, edgecolor='g', facecolor='none')
                # ax[0].add_patch(rect)
                draw_bbox_pred = compute_3d_box_vertices_withlist(cur_topk_bbox[j].numpy())  # [8, 3]
                draw_bbox_pred = project_3d_to_2d(draw_bbox_pred, intrinsic, extrinsic)  # [8, 2]
                draw_bbox_pred = standardize_bbox(draw_bbox_pred, H, W).numpy()
                edges = [(0,1),(1,2),(2,3),(3,0),
                        (4,5),(5,6),(6,7),(7,4),
                        (0,4),(1,5),(2,6),(3,7)]
                for edge in edges:
                    ax[0].plot(*zip(draw_bbox_pred[edge[0]], draw_bbox_pred[edge[1]]), color='g')
            if cur_topk_scores[j].item() > 0.6:
                # draw_bbox_pred = dataset_utils.recover_bbox(cur_topk_bbox[j], H, W)  # [4]
                # rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                #                          draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                #                          linewidth=1, edgecolor='b', facecolor='none')
                # ax[0].add_patch(rect)
                draw_bbox_pred = compute_3d_box_vertices_withlist(cur_topk_bbox[j].numpy())  # [8, 3]
                draw_bbox_pred = project_3d_to_2d(draw_bbox_pred, intrinsic, extrinsic)  # [8, 2]
                draw_bbox_pred = standardize_bbox(draw_bbox_pred, H, W).numpy()
                edges = [(0,1),(1,2),(2,3),(3,0),
                        (4,5),(5,6),(6,7),(7,4),
                        (0,4),(1,5),(2,6),(3,7)]
                for edge in edges:
                    ax[0].plot(*zip(draw_bbox_pred[edge[0]], draw_bbox_pred[edge[1]]), color='b')
            # plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tmp.png'))
            plt.close()
            frames.append(cv2.imread(os.path.join(output_dir, 'tmp.png'))[...,::-1])
        save_name = os.path.join(output_dir, '{}_{}.gif'.format(iter_num, i))
        imageio.mimsave(save_name, frames, 'GIF', duration=1)

def vis_feature(sample, pred, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    featrue_vis=pred['featrue_vis']

    cpu_featrue_vis = {key: tensor.cpu() for key, tensor in featrue_vis.items()}

    query_vis=cpu_featrue_vis['query_feat_after_extract_feature_vis']
    


    B,_,H,W=query_vis.shape

    for i in range(B):
        frames = []
        fig, ax = plt.subplots(2,4, dpi=100)
        for index, (key, value) in enumerate(cpu_featrue_vis.items()):                                                                          
            
            cur_feat_np,cur_channel=random_channel_vis(value[i])
            
            idxi=index//4
            idxj=index%4
            #These values ​​are then used to format the string
            title_str = '%s:channel %s' %(key,cur_channel)

            #Set title with corrected string
            
            ax[idxi,idxj].set_title(title_str,fontsize=5)

            ax[idxi,idxj].imshow(cur_feat_np,cmap='gray')

            plt.tight_layout()
            
        plt.tight_layout()    
        plt.savefig(os.path.join(output_dir, '{}_{}.png'.format(iter_num, i)))
        plt.close()

def random_channel_vis(image_tensor):
    '''tensor: shape c h w
    reture  h w
    '''
    channel_num=image_tensor.shape[0]
    # vis_channel=random.randint(0, channel_num - 1)
    vis_channel=2
    cur_image=image_tensor[vis_channel]
    cur_iamge_np = cur_image.numpy()
    return cur_iamge_np,vis_channel