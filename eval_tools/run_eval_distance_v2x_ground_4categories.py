#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: run_eval_distance_v2x_ground_4categories.py
author: yexiaoqing, liyingying
"""
import os
import time
import numpy
import logging
import multiprocessing
import math
from math import sin, cos, atan, sqrt, atan2
import numpy as np
from google.protobuf import text_format
import pdb
import config4cls as config
import pickle
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import argparse

def compute_min_dist(corners_bottom_res, corners_bottom_gt):
    """compute min distance
    input: [4, 3] vs [4, 3]
    """
    stepval = 1
    dist_points_res_gt = numpy.linalg.norm(corners_bottom_res[0:4:stepval, :]
                                           - corners_bottom_gt[0:4:stepval, :], axis=0)
    dist_gt_res = numpy.mean(dist_points_res_gt)
    corners_bottom_res0 = corners_bottom_res[0:4:stepval, :]
    corners_bottom_res_r1 = np.hstack([corners_bottom_res0[:, 1:], corners_bottom_res0[:, 0:1]])
    corners_bottom_res_r2 = np.hstack([corners_bottom_res0[:, 2:], corners_bottom_res0[:, 0:2]])
    corners_bottom_res_r3 = np.hstack([corners_bottom_res0[:, 3:], corners_bottom_res0[:, 0:3]])
    dist_gt_res = np.minimum(dist_gt_res,
                             np.mean(numpy.linalg.norm(corners_bottom_res_r1
                            - corners_bottom_gt[0: 4:stepval, :], axis = 0)))
    dist_gt_res = np.minimum(dist_gt_res,
                             np.mean(numpy.linalg.norm(corners_bottom_res_r2
                            - corners_bottom_gt[0: 4:stepval, :], axis = 0)))
    dist_gt_res = np.minimum(dist_gt_res,
                             np.mean(numpy.linalg.norm(corners_bottom_res_r3
                            - corners_bottom_gt[0: 4:stepval, :], axis = 0)))
    return dist_gt_res

def compute_similarity(list_res_dict):
    """compute similarity"""
    count = 0
    ACS_list = []
    AOS_list = []
    ASS_list = []
    AGD4_class_aware_list = []
    AGD4_abs_list = []
    AGD4_relative_list = []
    for ind, res in enumerate(list_res_dict):
        count += 1
        ACS_list.append(1 - res['relative_groundcenter'])
        AOS_list.append((1 + res['cos_orientation']) /2)
        ASS_list.append(1 - res['relative_LW'])
        AGD4_abs_list.append( res['delta_bottom4'])
        AGD4_relative_list.append( 1 - res['relative_bottom4'])
    ACS = np.sum(np.array(ACS_list)) / count
    AOS = np.sum(np.array(AOS_list)) / count
    ASS = np.sum(np.array(ASS_list)) / count
    AGD4_abs = np.sum(np.array(AGD4_abs_list)) / count
    AGD4_Q99 = np.percentile(np.array(AGD4_abs_list), 99)
    AGD4_Q90 = np.percentile(np.array(AGD4_abs_list), 90)
    AGD4_relative = np.sum(np.array(AGD4_relative_list)) / count
    
    error_txt = '{:6}\t{:6}\t{:6}\t{:6}\t{:6}\t{:6}\t{:6}\n{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}'.format(
	'ACS', 'AOS', 'ASS', 'AGD4_abs', 'AGD4_Q90', 'AGD4_Q99', 'AGD4_rel',
         ACS, AOS, ASS, AGD4_abs, AGD4_Q90, AGD4_Q99, AGD4_relative)
    
    return [error_txt, [ACS, AOS, ASS, AGD4_abs, AGD4_Q90, AGD4_Q99, AGD4_relative], ACS_list, AOS_list, ASS_list, AGD4_abs_list, AGD4_relative_list]

def project_3d(p2, location, hwl, ry3d, de_norm):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        location: x3d: x-coordinate of center of object
                  y3d: y-coordinate of center of object
                  z3d: z-cordinate of center of object
        hwl:h3d: height of object
            w3d: width of object
            l3d: length of object
        ry3d: rotation w.r.t y-axis
        de_norm: the ground equation plane

    """
    h3d = hwl[0]
    w3d = hwl[1]
    l3d = hwl[2]
    [x3d, y3d, z3d] = location
    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])
   
    x_corners = [l3d / 2, l3d / 2, -l3d / 2, -l3d / 2, l3d / 2, l3d / 2, -l3d / 2, -l3d / 2]
    y_corners = [0, 0, 0, 0, -h3d, -h3d, -h3d, -h3d]
    z_corners = [w3d / 2, -w3d / 2, -w3d / 2, w3d / 2, w3d / 2, -w3d / 2, -w3d / 2, w3d / 2]

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)
    R2 = np.array([[1, 0, 0],
                   [0, -de_norm[1], +de_norm[2]],
                   [0, -de_norm[2], -de_norm[1]]])
    corners_3d = R2.dot(corners_3d)
    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[0:2].T.tolist()
    
    for i in range(8):
        corners_2D[i][0] = int(corners_2D[i][0])
        corners_2D[i][1] = int(corners_2D[i][1])
   
    return corners_3d, corners_2D


def get_ground2camera(denorm):
    """input : denorm a, b, c, d
    output: r_ground2cam, t_ground2cam, r_cam2ground, t_cam2ground 
    """
    normal_vector = denorm[:3]
    d = denorm[-1]
    if normal_vector[1] > 0:
        normal_vector = -1 * normal_vector
        d = -1 * d
    new_z_axis = np.array(normal_vector)
    camera_x_axis = np.array([1, 0, 0])
    new_x_axis = camera_x_axis - camera_x_axis.dot(new_z_axis) * new_z_axis
    new_y_axis = np.cross(new_z_axis, new_x_axis)
    
    g2c_trans = np.zeros((3, 3), dtype=np.float32)
    g2c_trans[0, :] = new_x_axis
    g2c_trans[1, :] = new_y_axis
    g2c_trans[2, :] = new_z_axis
    g2c_trans_homo = np.zeros((4, 4), dtype=np.float32)
    g2c_trans_homo[:3, :3] = g2c_trans
    g2c_trans_homo[3, 3] = 1
    g2c_trans_homo[2, 3] = d
    c2g_trans_homo = np.linalg.inv(g2c_trans_homo)
    r_ground2cam = g2c_trans
    t_ground2cam = g2c_trans_homo[:3, 3:]
    r_cam2ground = c2g_trans_homo[:3, :3]
    t_cam2ground = c2g_trans_homo[:3, 3:] 
    return [r_ground2cam, t_ground2cam, r_cam2ground, t_cam2ground, g2c_trans_homo]


def read_kitti_denorm(denormfile):
    """
    Reads the v2x de_norm file from disc.

    Args:
        calfile (str): path to single de_norm file
    """
    text_file = open(denormfile, 'r')
    for line in text_file:
        parsed = line.split('\n')[0].split(' ')
        if parsed is not None and len(parsed) > 3:
            de_norm = []
            de_norm.append(float(parsed[0]))
            de_norm.append(float(parsed[1]))
            de_norm.append(float(parsed[2]))
            de_norm.append(float(parsed[3]))
    text_file.close()
    return de_norm

def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.
    Args:
        calfile (str): path to single calibration file
    """
    text_file = open(calfile, 'r')
    for line in text_file:
        parsed = line.split('\n')[0].split(' ')
        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None and parsed[0] == 'P2:':
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed[1]
            p2[0, 1] = parsed[2]
            p2[0, 2] = parsed[3]
            p2[0, 3] = parsed[4]
            p2[1, 0] = parsed[5]
            p2[1, 1] = parsed[6]
            p2[1, 2] = parsed[7]
            p2[1, 3] = parsed[8]
            p2[2, 0] = parsed[9]
            p2[2, 1] = parsed[10]
            p2[2, 2] = parsed[11]
            p2[2, 3] = parsed[12]
            p2[3, 3] = 1

    text_file.close()
    return p2

def parse_kitti_format(line):
    """Parse a string of kitti format into a dict for a object
    @param line: the string of kitti format
    @return dict_line: the output dict of the parsed string
    """
    v = line.strip().split()
    dict_line = {'type': v[0].lower(),
                 'truncated': float(v[1]),
                 'occluded': int(v[2]),
                 # observation angle
                 'alpha': float(v[3]),
                 # left, top, right, bottom
                 'bbox': [float(val) for val in v[4:8]],
                 # height, width, length
                 'size_hwl': [float(val) for val in v[8:11]],
                 # center
                 'location': [float(val) for val in v[11:14]],
                 # rot angle w.r.t y axis pith, 如何求yawl
                 'rotation_y': float(v[14])}
    if len(v) > 15:
        dict_line['score'] = float(v[15])
    if len(v) > 16:
        dict_line['id'] = int(v[16])
    if len(v) > 17:
        if len(v) > 20:
            dict_line['pts8'] = [float(val) for val in v[17:33]]
            dict_line['ry_cam'] = float(v[33])
        else:
            dict_line['ry_cam'] = float(v[17])
    else :
        dict_line['id'] = 0
    return dict_line

def calc_iou(obj1, obj2):
    """Calculate IoU of two objects
    @param obj1: the dict of first object
    @param obj2: the dict of second object
    @return iou: intersection-over-union (IOU) of two bounding boxes
    """
    x_min1 = obj1['bbox'][0]
    x_max1 = obj1['bbox'][2]
    y_min1 = obj1['bbox'][1]
    y_max1 = obj1['bbox'][3]
    x_min2 = obj2['bbox'][0]
    x_max2 = obj2['bbox'][2]
    y_min2 = obj2['bbox'][1]
    y_max2 = obj2['bbox'][3]
    if x_min1 >= x_max1 or y_min1 >= y_max1 or x_min2 >= x_max2 or y_min2 >= y_max2:
        return -1.0
    if x_max1 <= x_min2 or x_max2 <= x_min1 or y_max1 <= y_min2 or y_max2 <= y_min1:
        return 0.0
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    x_min_inter = max(x_min1, x_min2)
    x_max_inter = min(x_max1, x_max2)
    y_min_inter = max(y_min1, y_min2)
    y_max_inter = min(y_max1, y_max2)
    area_inter = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    iou = area_inter / (area1 + area2 - area_inter)
    return iou

def drawbox3d_8points_onImage(img, boxes2d_pred, boxes2d_gt, box3d_res_8points, box3d_gt8points, save_name, draw3d=False, res_pi=None):
    """draw 8 points on image""" 
    draw = ImageDraw.Draw(img)
    for index, box in enumerate(box3d_res_8points):
        if draw3d:
            box3d = np.array(box3d_res_8points[index])
            box3d = box3d.reshape((16))
            box3d_gt = np.array(box3d_gt8points[index])
            box3d_gt = box3d_gt.reshape((16))
            draw.line([box3d[0],box3d[1],box3d[2],box3d[3]],fill='green',width=2)
            draw.line([box3d[2],box3d[3],box3d[4],box3d[5]],fill='green',width=2)
            draw.line([box3d[4],box3d[5],box3d[6],box3d[7]],fill='green',width=2)
            draw.line([box3d[0],box3d[1],box3d[6],box3d[7]],fill='green',width=2)
            #4-5
            draw.line([box3d[8],box3d[9],box3d[10], box3d[11]],fill='red',width=2)
            #5-6
            draw.line([box3d[12],box3d[13],box3d[10],box3d[11]],fill='red',width=2)
            #6-7
            draw.line([box3d[12],box3d[13],box3d[14],box3d[15]],fill='red',width=2)
            #7-4
            draw.line([box3d[8],box3d[9],box3d[14],box3d[15]],fill='red',width=2)
            #vertical
            draw.line([box3d[0],box3d[1],box3d[8],box3d[9]],fill='red',width=2)
            draw.line([box3d[2],box3d[3],box3d[10],box3d[11]],fill='red',width=2)
            draw.line([box3d[4],box3d[5],box3d[12],box3d[13]],fill='red',width=2)
            draw.line([box3d[6],box3d[7],box3d[14],box3d[15]],fill='red',width=2)
            #plot gt
            draw.line([box3d_gt[0],box3d_gt[1],box3d_gt[2],box3d_gt[3]],fill='yellow',width=2)
            draw.line([box3d_gt[2],box3d_gt[3],box3d_gt[4],box3d_gt[5]],fill='yellow',width=2)
            draw.line([box3d_gt[4],box3d_gt[5],box3d_gt[6],box3d_gt[7]],fill='yellow',width=2)
            draw.line([box3d_gt[0],box3d_gt[1],box3d_gt[6],box3d_gt[7]],fill='yellow',width=2)
            #4-5
            draw.line([box3d_gt[8],box3d_gt[9],box3d_gt[10],box3d_gt[11]],fill='yellow',width=2)
            #5-6
            draw.line([box3d_gt[12],box3d_gt[13],box3d_gt[10],box3d_gt[11]],fill='yellow',width=2)
            #6-7
            draw.line([box3d_gt[12],box3d_gt[13],box3d_gt[14],box3d_gt[15]],fill='yellow',width=2)
            #7-4
            draw.line([box3d_gt[8],box3d_gt[9],box3d_gt[14],box3d_gt[15]],fill='yellow',width=2)
            #vertical
            draw.line([box3d_gt[0],box3d_gt[1],box3d_gt[8],box3d_gt[9]],fill='yellow',width=2)
            draw.line([box3d_gt[2],box3d_gt[3],box3d_gt[10],box3d_gt[11]],fill='yellow',width=2)
            draw.line([box3d_gt[4],box3d_gt[5],box3d_gt[12],box3d_gt[13]],fill='yellow',width=2)
            draw.line([box3d_gt[6],box3d_gt[7],box3d_gt[14],box3d_gt[15]],fill='yellow',width=2)
        if res_pi is not None:
            box3d = np.array(res_pi[index])
            box3d = box3d.reshape((16))
            draw.line([box3d[0],box3d[1],box3d[2],box3d[3]],fill='blue',width=2)
            draw.line([box3d[2],box3d[3],box3d[4],box3d[5]],fill='blue',width=2)
            draw.line([box3d[4],box3d[5],box3d[6],box3d[7]],fill='blue',width=2)
            draw.line([box3d[0],box3d[1],box3d[6],box3d[7]],fill='blue',width=2)
            #4-5
            draw.line([box3d[8],box3d[9],box3d[10], box3d[11]],fill='blue',width=2)
            #5-6
            draw.line([box3d[12],box3d[13],box3d[10],box3d[11]],fill='blue',width=2)
            #6-7
            draw.line([box3d[12],box3d[13],box3d[14],box3d[15]],fill='blue',width=2)
            #7-4
            draw.line([box3d[8],box3d[9],box3d[14],box3d[15]],fill='blue',width=2)
            #vertical
            draw.line([box3d[0],box3d[1],box3d[8],box3d[9]],fill='blue',width=2)
            draw.line([box3d[2],box3d[3],box3d[10],box3d[11]],fill='blue',width=2)
            draw.line([box3d[4],box3d[5],box3d[12],box3d[13]],fill='blue',width=2)
            draw.line([box3d[6],box3d[7],box3d[14],box3d[15]],fill='blue',width=2)

    img.save(save_name)

def calc_relative_error_of_objects_Surver_ground(res, gt, intrinsics, detail_file,
                                r_g2cam, t_g2cam, r_cam2g, t_cam2g, 
                                g2c_trans, de_norm):
    """Calculate the relative error of the 3D location result and the ground truth
    
    """
    dist_gt_ori = sqrt((gt['location'][0] ** 2 + gt['location'][2] ** 2)) #x^2 + z^2
    c2g_trans = np.linalg.inv(g2c_trans)
    def get_8_coord(bbox_3d):
        box_3d_coord = []
        for i in range(8):
            box_3d_coord.append([bbox_3d[i]['x'],bbox_3d[i]['y']])
        return box_3d_coord 
   
    if res['type'] == 'pedestrian':
        res['rotation_y'] = gt['rotation_y']

    corners_bottom_res, corners_2d_pred_new = project_3d(intrinsics, res['location'], res['size_hwl'], res['rotation_y'], de_norm)
    corners_bottom_gt, corners_2d_gt_new = project_3d(intrinsics, gt['location'], gt['size_hwl'], gt['rotation_y'], de_norm)
    #convert to ground plane
    corners_bottom_res = corners_bottom_res.T[:4, :]
    corners_bottom_gt = corners_bottom_gt.T[:4, :]
    dist_gt_res = compute_min_dist(corners_bottom_res, corners_bottom_gt)
    
    if abs(res['rotation_y'] - gt['rotation_y']) > math.pi / 2.0 :
        if res['rotation_y'] >=0 :
            res['rotation_y'] += math.pi
        else :
            res['rotation_y'] += -math.pi
    corners_bottom_res_pi, corners_2d_pred_new_pi = project_3d(intrinsics, res['location'], res['size_hwl'], res['rotation_y'], de_norm)
    corners_bottom_res_pi = corners_bottom_res_pi.T[:4, :]
    dist_gt_res_pi = compute_min_dist(corners_bottom_res_pi, corners_bottom_gt)
    dist_gt_res = np.minimum(dist_gt_res, dist_gt_res_pi)

    relative_error = dist_gt_res / (dist_gt_ori + 1e-7) #平均误差除以深度距离
    detail = [res['id'], res['type'], res['bbox'], res['location'], gt['id'], gt['type'], gt['bbox'], gt['location'], dist_gt_res, dist_gt_ori, relative_error]
    if config.debug_flag:
        fp = open(detail_file, 'a')
        fp.write(str(detail) + '\n')
        fp.close()
    box3d_pred = None
    box3d_gt = None
    res_dict = {}
    res_dict['delta_groundcenter'] = np.linalg.norm(np.array(gt['location']) - np.array(res['location']))
    res_dict['relative_groundcenter'] = np.minimum(1.0, res_dict['delta_groundcenter'] / np.linalg.norm(gt['location']))
    res_dict['delta_orientation'] = np.minimum(np.fabs(gt['rotation_y'] - res['rotation_y']), 2 * np.pi - np.fabs(gt['rotation_y'] - res['rotation_y']))
    res_dict['cos_orientation'] =  (1.0 + cos(res_dict['delta_orientation'])) / 2.0
    res_dict['delta_LW'] = np.fabs(gt['size_hwl'][1] * gt['size_hwl'][2] - res['size_hwl'][1] * res['size_hwl'][2])
    res_dict['relative_LW'] = np.minimum(1.0, res_dict['delta_LW'] / (gt['size_hwl'][1] * gt['size_hwl'][2]))
    res_dict['delta_bottom4'] = dist_gt_res
    res_dict['relative_bottom4'] = np.minimum(1.0, dist_gt_res / (dist_gt_ori + 1e-7))
    res_dict['pred_gt_type'] = [res['type'], gt['type']]
    return [relative_error, dist_gt_ori, dist_gt_res, corners_2d_pred_new_pi, corners_2d_pred_new, corners_2d_gt_new], res_dict

def calc_relative_error_of_files_Surver_ground(file_res, file_gt, intrinsic, detail_file, denorm,
                                 set_type=['car'],
                                 iou_thres=0.5, score_thres=0, h_thres=1):
    """Calculate the relative error of each object in a file
    @param file_res: the file of the results of the objects
    @param file_gt: the file of the ground truth of the objects
    @param intrinsic: the intrinsic matrix 
    @param detail_file: the detail log file
    @param denorm: the ground equation plane
    @param set_type: the set of types for evaluation
    @param iou_thres: the threshold for the IoU of two objects (default: 0.7)
    @param score_thres: the threshold for the dectection score or confidence (default: 0.6)
    @param h_thres: the threshold for the height of the object (default: 1.0 meter)
    @return list_rerr_dist: the list of the pair of the relative error and the distance
    """
    allcount = 0
    
    r_ground2cam, t_ground2cam, r_cam2ground, t_cam2ground, g2c_trans = get_ground2camera(denorm)
    
    list_res = open(file_res).readlines()
    list_res = [parse_kitti_format(line) for line in list_res]
    list_gt = open(file_gt).readlines()
    list_gt = [parse_kitti_format(line) for line in list_gt]

    list_rerr_dist = []
    list_res_dict = []
    sublist_res = [res for res in list_res if (
        res['type'] in set_type and res['size_hwl'][0] > h_thres
        and res['location'][2] > 0)]
    
    sublist_gt = []
    sublist_anno = []
    for ind, gt in enumerate(list_gt):
        if (gt['type'] in set_type and gt['size_hwl'][0] > h_thres and gt['location'][2]>0):
            sublist_gt.append(gt)
    
    
    if len(sublist_res) > 0 and len(sublist_gt) > 0:
        table_iou = [[calc_iou(res, gt) for gt in sublist_gt]
                     for res in sublist_res]
        for i in range(len(sublist_res)):
            iou_max = max(table_iou[i][:])
            if iou_max > float(iou_thres):
                ind_max = table_iou[i][:].index(iou_max)
                col = [row[ind_max] for row in table_iou]
                iou_max_inv = max(col)
                ind_max_inv = col.index(iou_max_inv)
                if i == ind_max_inv and sublist_res[i]['type'] == sublist_gt[ind_max]['type']:
                    rt, res_dict = calc_relative_error_of_objects_Surver_ground(
                        sublist_res[i], sublist_gt[ind_max], intrinsic, 
                        detail_file, r_ground2cam, t_ground2cam, 
                        r_cam2ground, t_cam2ground, g2c_trans, denorm)
                    
                    list_rerr_dist.append(rt)
                    list_res_dict.append(res_dict)
                    allcount += 1
                    
    return list_rerr_dist, allcount, list_res_dict

def evaluate_SurveillanceCamera(eval_range_min=0, eval_range_max=120, eval_range_step=30, log_file='result.txt', 
                                set_type=['vehicle', 'car', 'van', 'bus', 'truck'],
                                coarse_label_path=None, iou_thresh=0.5):
    """print the table of the relative error of the tracked dectection results
    @param eval_range_min: the lower bound of the evaluation range
    @param eval_range_max: the upper bound of the evaluation range
    @param eval_range_step: the step of the evaluation range
    """

    os.system('mkdir -p %s' % (config.debug_dir))
    detail_file = '%s/detail.txt' % (config.debug_dir)
    folder_seq_res = "%s/" % (config.transform_detection_dir)
    folder_seq_gt = '%s/' % (config.label_dir)
    if coarse_label_path is not None:
        folder_seq_gt = coarse_label_path + "/label_2"
    list_seq = os.listdir(folder_seq_res) 
    list_seq.sort()
    error_file = log_file 
    image_dir  = "%s/" % (config.image_root)

    list_rerr_dist = []
    list_similarity = []
    allsum = 0
    increase = 0
    for i, name in enumerate(list_seq):
        name_key = name.strip()
        name_key = name_key.split("/")[-1]
        name_key = name_key.split(".")[0]
        name_key = name_key.replace(".jpg", "")
        anno_this = None
        intrinsic = read_kitti_cal(os.path.join(config.intrinsic_dir, name_key + '.txt'))
        intrinsic = numpy.matrix(intrinsic).reshape(4, 4)

        denorm = read_kitti_denorm(os.path.join(config.denorm_dir, name_key + '.txt'))  
        file_gt = os.path.join(folder_seq_gt, '%s.txt' % name_key.strip())
        file_res = os.path.join(folder_seq_res, '%s.txt' % name_key.strip())
        if not os.path.exists(file_gt): continue

        list_rerr_dist_file, allcount, list_res_dict = calc_relative_error_of_files_Surver_ground(
            file_res, file_gt, intrinsic, detail_file, denorm, set_type=set_type, iou_thres=iou_thresh)
        list_rerr_dist = list_rerr_dist + list_rerr_dist_file
        list_similarity = list_similarity + list_res_dict
        
        allsum += allcount
        increase += allcount
        if (increase >= 10000):
           print("already computed {} samples".format(allsum))
           increase = 0
    print("--------------allsum: ", allsum)
    if allsum < 1: return
    [similarity_txt, [ACS, AOS, ASS, AGD4_abs, AGD4_Q90, AGD4_Q99, AGD4_relative],
     ACS_list, AOS_list, ASS_list, AGD4_abs_list, AGD4_relative_list] = compute_similarity(list_similarity)

    list_rerr_all = [rerr_dist[0] for rerr_dist in list_rerr_dist
                     if rerr_dist[1] > eval_range_min and rerr_dist[1] <= eval_range_max]#0 is relative error
    list_rerr_all_1 = [rerr_dist[1] for rerr_dist in list_rerr_dist
                     if rerr_dist[1] > eval_range_min and rerr_dist[1] <= eval_range_max]#1 is sqrt(x^2 + y^2), distance
    list_rerr_all_2 = [rerr_dist[2] for rerr_dist in list_rerr_dist
                     if rerr_dist[1] > eval_range_min and rerr_dist[1] <= eval_range_max]#2 is 4points average error
    list_all_ACS = [] 
    list_all_AOS = [] 
    list_all_ASS = [] 
    list_all_AGD4_abs = [] 
    list_all_AGD4_relative = [] 

    valid_index_list = []
    for valid_index, dist in enumerate(list_rerr_dist):
        if dist[1] > eval_range_min and dist[1] < eval_range_max:
            valid_index_list.append(valid_index)
            list_all_ACS.append(ACS_list[valid_index])
            list_all_AOS.append(AOS_list[valid_index])
            list_all_ASS.append(ASS_list[valid_index])
            list_all_AGD4_abs.append(AGD4_abs_list[valid_index])
            list_all_AGD4_relative.append(AGD4_relative_list[valid_index])

    list_range = range(eval_range_min, eval_range_max, eval_range_step)
    list_all_ACS_var_range = [[], [], [], []]
    list_all_AOS_var_range = [[], [], [], []] # * len(list_range) 
    list_all_ASS_var_range = [[], [], [], []] # * len(list_range)
    list_all_AGD4_abs_var_range = [[], [], [], []] # * len(list_range)
    list_all_AGD4_relative_var_range = [[], [], [], []] # * len(list_range) 
    #pdb.set_trace()
    for valid_index, dist in enumerate(list_rerr_dist):
        if dist[1] > list_range[0] and dist[1] <= list_range[0] + eval_range_step:
            cur_ind = 0
        elif dist[1] > list_range[1] and dist[1] <= list_range[1] + eval_range_step:
            cur_ind = 1
        elif dist[1] > list_range[2] and dist[1] <= list_range[2] + eval_range_step:
            cur_ind = 2
        elif dist[1] > list_range[3] and dist[1] <= list_range[3] + eval_range_step:
            cur_ind = 3
        else: continue
        list_all_ACS_var_range[cur_ind].append(ACS_list[valid_index])
        list_all_AOS_var_range[cur_ind].append(AOS_list[valid_index])
        list_all_ASS_var_range[cur_ind].append(ASS_list[valid_index])
        list_all_AGD4_abs_var_range[cur_ind].append(AGD4_abs_list[valid_index])
        list_all_AGD4_relative_var_range[cur_ind].append(AGD4_relative_list[valid_index])
    
    plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    colors = numpy.random.rand(len(list_rerr_all))
    plt.sca(ax1)
    plt.scatter(list_rerr_all_1, list_rerr_all, c=colors, marker='.')
    plt.xlabel("distance/m")
    plt.ylabel("relative-error/%")
    plt.sca(ax2)
    plt.scatter(list_rerr_all_1, list_rerr_all_2, c=colors, marker='.')
    plt.xlabel("distance/m")
    plt.ylabel("abs-error/m")
    plt.savefig('%s/result_img.png' % (config.debug_dir))
    # plt.show()
    
    print("set_type for 4 categories: ", set_type)
    R_score = '-'
    type_txt = '''
-------------------------set type in 4: {}--------------------------------------
'''.format(set_type[0])
    error_txt = '''
==============================================================
Table of Relative Error
==============================================================
range\ttotal\tR_s\tACS\tAOS\tASS\tAGD-rel\tAdded\tAGD4-90\tAGD-99\tAGD4-m\tmean-m\tQ99-m\t
all\t{:d}\t{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}
    '''.format(len(list_rerr_all),
    R_score,
    ACS, 
    AOS, 
    ASS, 
    AGD4_relative,
    (ACS + AOS + ASS + AGD4_relative) / 4.0,
    AGD4_abs, 
    AGD4_Q90, 
    AGD4_Q99, 
    numpy.mean(list_rerr_all_2),
    numpy.percentile(list_rerr_all_2,99)
)
    print(error_txt)
    with open (error_file, 'a') as ef :
        ef.write(type_txt)
        ef.write(error_txt)
        ef.write('\n')
        list_range = range(eval_range_min, eval_range_max, eval_range_step)
        for cur_ind, range_start in enumerate(list_range):
            range_end = range_start + eval_range_step
            list_rerr_range = [rerr_dist[0] for rerr_dist in list_rerr_dist
                               if rerr_dist[1] > range_start and rerr_dist[1] <= range_end]
            list_rerr_range2 = [rerr_dist[2] for rerr_dist in list_rerr_dist
                               if rerr_dist[1] > range_start and rerr_dist[1] <= range_end]

            if len(list_rerr_range) < 1:
                continue
            area_error_txt = '%d-%d\t%d\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (
            range_start, range_end, \
            len(list_rerr_range), \
            R_score, 
            np.sum(np.array(list_all_ACS_var_range[cur_ind])) / len(list_all_ACS_var_range[cur_ind]),
            np.sum(np.array(list_all_AOS_var_range[cur_ind])) / len(list_all_AOS_var_range[cur_ind]),
            np.sum(np.array(list_all_ASS_var_range[cur_ind])) / len(list_all_ASS_var_range[cur_ind]),
            np.sum(np.array(list_all_AGD4_relative_var_range[cur_ind])) / len(list_all_AGD4_relative_var_range[cur_ind]),
            np.sum(np.array(list_all_ACS_var_range[cur_ind] + list_all_AOS_var_range[cur_ind] + 
               list_all_ASS_var_range[cur_ind] + list_all_AGD4_relative_var_range[cur_ind])) / len(list_all_ACS_var_range[cur_ind]) / 4.0,
            np.sum(np.array(list_all_AGD4_abs_var_range[cur_ind])) / len(list_all_AGD4_abs_var_range[cur_ind]),
            np.percentile(list_all_AGD4_abs_var_range[cur_ind], 90),
            np.percentile(list_all_AGD4_abs_var_range[cur_ind], 99),
            numpy.mean(list_rerr_range2), \
            numpy.percentile(list_rerr_range2, 99))
            print(area_error_txt)
            ef.write(area_error_txt)
            ef.write('\n')
        ef.write('==============================================================')
        print('==============================================================')

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--log_file', default='result.txt',
                         help='log_file')
    argparser.add_argument('--coarse_label_path', default=None,
                         help='coarse_label_path')
    argparser.add_argument('--result_path', default=None,
                         help='result_path')
    argparser.add_argument('--iou_thresh', default=None,
                         help='iou_thresh')
    args = argparser.parse_args()
    config.transform_detection_dir = args.result_path
    evaluate_SurveillanceCamera(log_file=args.log_file, set_type=['car'],
                                coarse_label_path=args.coarse_label_path,
                                iou_thresh=args.iou_thresh)
    evaluate_SurveillanceCamera(log_file=args.log_file, set_type=['big_vehicle'],
                                coarse_label_path=args.coarse_label_path,
                                iou_thresh=args.iou_thresh)
    evaluate_SurveillanceCamera(log_file=args.log_file, set_type=['cyclist'],
                                coarse_label_path=args.coarse_label_path,
                                iou_thresh=args.iou_thresh)
    evaluate_SurveillanceCamera(log_file=args.log_file, set_type=['pedestrian'],
                                coarse_label_path=args.coarse_label_path,
                                iou_thresh=args.iou_thresh)
