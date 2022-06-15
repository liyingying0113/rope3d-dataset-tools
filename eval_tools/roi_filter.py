#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" 
File: roi_filter.py
author: shumao, liyingying
"""
import json
import argparse
import pdb
import os
import sys
import cv2
from PIL import Image
import glob
import config4cls as config

parser = argparse.ArgumentParser(description = "convert parameters")
parser.add_argument("--txt_dir", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

txt_path = args.txt_dir
output_dir = args.output_dir
calib_dir = config.intrinsic_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("outputdir:", output_dir) 

print('doing roi filter ...')
files = os.listdir(txt_path)
for f in files:
    calib_path = os.path.join(calib_dir, f)
    calib = open(calib_path).readlines()[0].split(' ')[1]
    
    img_mask_path = glob.glob("./mask/" + calib + '*.jpg')[0]
    
    img_mask = Image.open(img_mask_path)
    part = f.split('_')
    output_path = os.path.join(output_dir, f)
    fh1 = open(os.path.join(txt_path, f), "r")

    file = open(output_path, 'w')
    file.close()

    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
 
        xmin = float(splitLine[4])
        ymin = float(splitLine[5])
        xmax = float(splitLine[6])
        ymax = float(splitLine[7])
        center_x = int((xmin+xmax) / 2)
        center_y = int((ymin+ymax) / 2)
        
        if center_x >= img_mask.width or center_y >= img_mask.height or center_y < 200:
            continue
        if img_mask.getpixel((center_x, center_y)) == (255, 255, 255):
            with open(output_path, 'a') as fp:
                fp.write("{}\n".format(line))
print('Done')

