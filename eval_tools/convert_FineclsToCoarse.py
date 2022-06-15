""""""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: convert_fine2coarse.py
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
import pickle
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
    else:
        dict_line['id'] = 0
    return dict_line




def convert_fine2coarse(coarse_gt_path="./"):
    """Parse 9cls label to 4cls label
    @param coarse_gt_path: the new 4cls label path
    """
    
    folder_seq_gt = '%s/' % (config.label_dir_9cls)
    print(folder_seq_gt)
    list_seq = os.listdir(folder_seq_gt) 

    list_rerr_dist = []

    allsum = 0
    validsum = 0
    fine2coarse = {}
    fine2coarse['van'] = 'car'
    fine2coarse['car'] = 'car'
    fine2coarse['bus'] = 'big_vehicle'
    fine2coarse['truck'] = 'big_vehicle'
    fine2coarse['cyclist'] = 'cyclist'
    fine2coarse['motorcyclist'] = 'cyclist'
    fine2coarse['tricyclist'] = 'cyclist'
    fine2coarse['pedestrian'] = 'pedestrian'
    fine2coarse['barrow'] = 'pedestrian' 

    if not os.path.exists(coarse_gt_path):
        print("mkdir")
        os.mkdir(coarse_gt_path)

    coarse_gt_path = coarse_gt_path + "/label_2"
    if not os.path.exists(coarse_gt_path):
        os.mkdir(coarse_gt_path)
    
    for i, name in enumerate(list_seq):
        name_key = name.strip()
        name_key = name_key.split("/")[-1]
        anno_this = None

        file_gt = os.path.join(folder_seq_gt, '%s' % name.strip())
        
        if not os.path.exists(file_gt): continue
        list_gt = open(file_gt).readlines()
        
        list_gt_new = []
        for ind, gt in enumerate(list_gt):
            type = gt.split(" ")[0]
            if (type in fine2coarse.keys()):
                gt_new = copy.deepcopy(gt.split(" "))
                gt_new[0] = fine2coarse[type]
                gt_new = " ".join(gt_new)
                list_gt_new.append(gt_new)

        with open(os.path.join(coarse_gt_path, '%s' % name.strip()), "w") as f:
            f.writelines(list_gt_new)






if __name__ == '__main__':
    config_util.update(config)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--coarse_label_path', default='coarse_gt_label',
                         help='coarse_new_gt_label_path')
    args = argparser.parse_args()
    convert_fine2coarse(coarse_gt_path=args.coarse_label_path)
