"""
File: show_groundpoints.py
project the ground points onto the image
author: yexiaoqing, liyingying
"""

import os
import cv2
import numpy as np
import math
import sys
import config
import glob as gb
import yaml
from pyquaternion import Quaternion

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


def read_kitti_ext(extfile):
    """read extrin"""
    text_file = open(extfile, 'r')
    cont = text_file.read()
    x = yaml.safe_load(cont)
    r = x['transform']['rotation']
    t = x['transform']['translation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    m = q.rotation_matrix
    m = np.matrix(m).reshape((3, 3))
    t = np.matrix([t['x'], t['y'], t['z']]).T
    p1 = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))
    points_on_ground = x['points_lane_detection']
    points_on_ground = np.array(points_on_ground).reshape(-1, 3) #[N, 3]
    return np.array(p1.I), points_on_ground


def progress(count, total, status=''):
    """ update a prograss bar"""
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def show_groundpoints(name_list):
    """ project the 3D ground points onto the image and show """
    #os.system('mkdir -p %s' % out_dir)
    image_root = config.image_dir
    extrinsics_dir = config.extrinsics_dir
    out_dir = config.out_points_dir
    cal_dir = config.cal_dir

    for i, name in enumerate(name_list):
        img_path = os.path.join(image_root, name)
        name = name.strip()
        name = name.split('/')
        name = name[-1].split('.')[0]
        progress(i, len(name_list))

        calfile = os.path.join(cal_dir, '%s.txt' % (name))
        p2 = read_kitti_cal(calfile)

        img = cv2.imread(img_path)
        h, w, c = img.shape

        yaml_data = yaml.safe_load(open(os.path.join(extrinsics_dir, '%s.yaml' % (name))))
        extrinsic_file = os.path.join(extrinsics_dir, '%s.yaml' % (name))
        world2camera, points_on_ground = read_kitti_ext(extrinsic_file)

        camera2world = np.linalg.inv(world2camera.reshape((4, 4))).reshape(4, 4)
        points_on_ground = points_on_ground.T #[3, N]
        

        #convert points_on_ground to camera-coord
        ones = np.ones(points_on_ground.shape[1]).reshape(1, -1).tolist() #[1, N]
        points_on_ground_homo = points_on_ground.tolist() + ones  #[4, N]
        points_in_camera = world2camera * np.matrix(points_on_ground_homo)
        
        points_2d = p2.dot(points_in_camera)
        points_2d = points_2d[:3, :]
        points_2d = points_2d / points_2d[2]  #[3, N]
        #pdb.set_trace()
        for ind in range(points_2d.shape[1]):
            point = (int(points_2d[0, ind]), int(points_2d[1, ind]))
            cv2.circle(img, point, 2, (0, 255, 0), 3) #pointsize = 2, point_color = (255, 0, 0), thickness = 3
        cv2.imwrite('%s/%s.jpg' % (out_dir, name), img)


if __name__ == '__main__':
    if config.val_list is None:
        name_list = gb.glob(config.extrinsics_dir + "/*")
    else:
        val_part_list  = open(config.val_list).readlines()
        name_list = []
        for name in val_part_list:
            name_list.append(name.split('\n')[0] + '.jpg')
        name_list.sort()

    if not os.path.isdir(config.out_points_dir):
        os.makedirs(config.out_points_dir)

    show_groundpoints(name_list)
