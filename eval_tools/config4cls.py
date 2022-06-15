#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: config.py
4cls
"""
import os
import time


current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
current_time = current_time.replace(' ', '-')

# dataset dir info
data_root = './data/v2x/validation/'
image_root = '%s/image_2' % data_root
label_dir_9cls = '%s/label_2' % data_root
label_dir = "./coarse4_label_path/label_2_filter" 

intrinsic_dir = '%s/calib' % data_root 
denorm_dir = '%s/denorm' % data_root 


transform_detection_dir = ""

debug_flag = True
debug_dir = "debug"
if not os.path.exists(debug_dir):
    os.mkdir(debug_dir)

