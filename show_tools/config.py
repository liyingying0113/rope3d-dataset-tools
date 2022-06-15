#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: config.py
"""


image_dir ='./data_demo/image_2'
cal_dir   ='./data_demo/calib'
label_dir='./data_demo/label_2'
denorm_dir='./data_demo/denorm'
out_box_dir   ='./vis_box/'
val_list  ='./data_demo/demo.txt'
extrinsics_dir = './data_demo/extrinsics'
out_points_dir   ='./vis_ground/'

color_list = {'car': (0, 0, 255),
              'truck': (0, 255, 255),
              'van': (255, 0, 255),
              'bus': (255, 255, 0),
              'cyclist': (0, 128, 128),
              'motorcyclist': (128, 0, 128),
              'tricyclist': (128, 128, 0),
              'pedestrian': (0, 128, 255),
              'barrow': (255, 0, 128)}

thresh = -0.5
