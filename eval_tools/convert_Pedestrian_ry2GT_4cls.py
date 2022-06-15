#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: convert_Pedestrian_ry2GT_4cls.py
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
import config_util
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import argparse
import copy

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

def convert_ry2gt(iou_thres = 0.5):
    """
    convert the ry of Pedestrian to GT value.
    @param iou_thres: the iou threshold of pred and gt, used to match
    """

    os.system('mkdir -p %s' % (config.debug_dir))
    
    folder_seq_res = "%s/" % (config.transform_detection_dir)
    folder_seq_gt = '%s/' % (config.label_dir)
    list_seq = os.listdir(folder_seq_gt) 
    last_name = folder_seq_res.split("/")[-1]
    
    if len(last_name) == 0:
        last_name = folder_seq_res.split("/")[-2]
    pred_ped_path = folder_seq_res.replace(last_name, last_name + "_ped")
    print("new pred pedestrian path: ", pred_ped_path)

    if not os.path.exists(pred_ped_path):
        os.mkdir(pred_ped_path)

    for i, name in enumerate(list_seq):
        name_key = name.strip()
        name_key = name_key.split("/")[-1]
        name_key = name_key.replace(".jpg", "")
        anno_this = None

        file_gt = os.path.join(folder_seq_gt, '%s' % name.strip())#label
        file_res = os.path.join(folder_seq_res, '%s' % name.strip())#pred
        if not os.path.exists(file_gt): continue
        list_res_ori = open(file_res).readlines()
        list_res = [parse_kitti_format(line) for line in list_res_ori]
        list_gt_ori = open(file_gt).readlines()
        list_gt = [parse_kitti_format(line) for line in list_gt_ori]
        list_res_new = []

        table_iou = [[calc_iou(res, gt) for gt in list_gt]
                     for res in list_res]
        
        for i in range(len(list_res)):
            try:
                iou_max = max(table_iou[i][:])
            except:
                continue
            flag = 0
            if iou_max > iou_thres:
                ind_max = table_iou[i][:].index(iou_max)
                col = [row[ind_max] for row in table_iou]
                iou_max_inv = max(col)
                ind_max_inv = col.index(iou_max_inv)
                if i == ind_max_inv and list_res[i]['type'] == list_gt[ind_max]['type'] and list_res[i]['type'] == 'pedestrian':
                    flag = 1
            cur_pred = list_res_ori[i].split(" ")
            if flag == 1:
                cur_gt = list_gt_ori[ind_max].split(" ")
                cur_pred[3] = cur_gt[3] # alpha
                cur_pred[14] = cur_gt[14].strip() # ry
                pred_new = " ".join(cur_pred)
                list_res_new.append(pred_new)
            else:
                pred_new = " ".join(cur_pred)
                list_res_new.append(pred_new)
        
        with open(os.path.join(pred_ped_path, '%s' % name.strip()), "w") as f:
            f.writelines(list_res_new)

if __name__ == '__main__':
    config_util.update(config)
    argparser = argparse.ArgumentParser()
   
    argparser.add_argument('--result_path', default=None,
                         help='result_path')
    args = argparser.parse_args()
    config.transform_detection_dir = args.result_path
    convert_ry2gt()
