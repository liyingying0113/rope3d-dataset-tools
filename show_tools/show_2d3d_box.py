"""
File: show_2d3d_box.py
author: yexiaoqing, liyingying
"""
import os
import cv2
import numpy as np
import math
import sys
import config
import glob as gb
import pdb
import yaml
from pyquaternion import Quaternion
# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)


class Data:
    """ class Data """
    def __init__(self, obj_type="unset", truncation=-1, occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1, \
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, detect_id=-1, \
                 vx=0, vy=0, vz=0):
        """init object data"""
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.detect_id = detect_id

    def __str__(self):
        """ str """
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


def progress(count, total, status=''):
    """ update a prograss bar"""
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

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


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, de_norm):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
        de_norm: [a,b,c] denotes the ground equation plane
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

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

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T
    return verts3d


def load_detect_data(filename):
    """
    load detection data of kitti format
    """
    data = []
    with open(filename) as infile:
        index = 0
        for line in infile:
            # KITTI detection benchmark data format:
            # (objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            line = line.strip()
            fields = line.split(" ")
            t_data = Data()
            # get fields from table
            t_data.obj_type = fields[
                0].lower()  # object type [car, pedestrian, cyclist, ...]
            t_data.truncation = float(fields[1])  # truncation [0..1]
            t_data.occlusion = int(float(fields[2]))  # occlusion  [0,1,2]
            t_data.obs_angle = float(fields[3])  # observation angle [rad]
            t_data.x1 = int(float(fields[4]))  # left   [px]
            t_data.y1 = int(float(fields[5]))  # top    [px]
            t_data.x2 = int(float(fields[6]))  # right  [px]
            t_data.y2 = int(float(fields[7]))  # bottom [px]
            t_data.h = float(fields[8])  # height [m]
            t_data.w = float(fields[9])  # width  [m]
            t_data.l = float(fields[10])  # length [m]
            t_data.X = float(fields[11])  # X [m]
            t_data.Y = float(fields[12])  # Y [m]
            t_data.Z = float(fields[13])  # Z [m]
            t_data.yaw = float(fields[14])  # yaw angle [rad]
            if len(fields) >= 16:
              t_data.score = float(fields[15])  # detection score
            else:
              t_data.score = 1
            t_data.detect_id = index
            data.append(t_data)
            index = index + 1
    return data


def load_denorm_data(denormfile):
    """
    Reads the v2x de_norm file from disc.

    Args:
        denormfile (str): path to single de_norm file
    """
    text_file = open(denormfile, 'r')
    for line in text_file:
        parsed = line.split('\n')[0].split(' ')
        if parsed is not None and len(parsed) > 3:
            de_norm = []
            de_norm.append(float(parsed[0]))
            de_norm.append(float(parsed[1]))
            de_norm.append(float(parsed[2]))
    text_file.close()

    return np.array(de_norm)


def project_3d_ground(p2, de_bottom_center, w3d, h3d, l3d, ry3d, de_norm, c2g_trans):
    """
    Projects a 3D box into 2D vertices using the camera2ground tranformation
    Note: Since the roadside camera contains pitch and roll angle w.r.t. the ground/world,
    simply adopting KITTI-style projection not works. We first compute the 3D bounding box in ground-coord and then convert back to camera-coord.

    Args:
        p2 (nparray): projection matrix of size 4x3
        de_bottom_center: bottom center XYZ-coord of the object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
        de_norm: [a,b,c] denotes the ground equation plane
        c2g_trans: camera_to_ground translation
    """
    de_bottom_center_in_ground = c2g_trans.dot(de_bottom_center) #convert de_bottom_center in Camera coord into Ground coord
    bottom_center_ground = np.array(de_bottom_center_in_ground)
    bottom_center_ground = bottom_center_ground.reshape((3, 1))
    theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
    theta0 = c2g_trans[:3, :3] * theta #first column
    yaw_world_res = math.atan2(theta0[1], theta0[0])
    g2c_trans = np.linalg.inv(c2g_trans)

    verts3d = get_camera_3d_8points_g2c(w3d, h3d, l3d, yaw_world_res,
        bottom_center_ground, g2c_trans, p2, isCenter=False)
    verts3d = np.array(verts3d)
    return verts3d


def project_3d_world(p2, de_center_in_world, w3d, h3d, l3d, ry3d, camera2world):
    """
    help with world
    Projects a 3D box into 2D vertices using the camera2world tranformation
    Note: Since the roadside camera contains pitch and roll angle w.r.t. the ground/world,
    simply adopting KITTI-style projection not works. We first compute the 3D bounding box in ground-coord and then convert back to camera-coord.

    Args:
        p2 (nparray): projection matrix of size 4x3
        de_bottom_center: bottom center XYZ-coord of the object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
        camera2world: camera_to_world translation
    """
    center_world = np.array(de_center_in_world) #bottom center in world
    theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
    theta0 = camera2world[:3, :3] * theta  #first column
    world2camera = np.linalg.inv(camera2world)
    yaw_world_res = math.atan2(theta0[1], theta0[0])
    verts3d = get_camera_3d_8points_g2c(w3d, h3d, l3d,
        yaw_world_res, center_world[:3, :], world2camera, p2, isCenter=False)

    verts3d = np.array(verts3d)
    return verts3d


def compute_c2g_trans(de_norm):
    """
    compute_c2g_trans
    author: yexiaoqing
    function: compute the camera to ground transformation (only rotation matrix)
    """
    #pdb.set_trace()
    ground_z_axis = de_norm 
    cam_xaxis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
    ground_x_axis = ground_x_axis / np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis = ground_y_axis / np.linalg.norm(ground_y_axis)
    c2g_trans = np.vstack([ground_x_axis, ground_y_axis, ground_z_axis]) #(3, 3)
    return c2g_trans


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
    return np.array(p1.I)


def get_camera_3d_8points_g2c(w3d, h3d, l3d, yaw_ground, center_ground,
                          g2c_trans, p2,
                          isCenter=True):
    """
    function: projection 3D to 2D
    w3d: width of object
    h3d: height of object
    l3d: length of object
    yaw_world: yaw angle in world coordinate
    center_world: the center or the bottom-center of the object in world-coord
    g2c_trans: ground2camera / world2camera transformation
    p2: projection matrix of size 4x3 (camera intrinsics)
    isCenter:
        1: center,
        0: bottom
    """
    ground_r = np.matrix([[math.cos(yaw_ground), -math.sin(yaw_ground), 0],
                         [math.sin(yaw_ground), math.cos(yaw_ground), 0],
                         [0, 0, 1]])
    #l, w, h = obj_size
    w = w3d
    l = l3d
    h = h3d

    if isCenter:
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                  [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    else:#bottom center, ground: z axis is up
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                  [0, 0, 0, 0, h, h, h, h]])

    corners_3d_ground = np.matrix(ground_r) * np.matrix(corners_3d_ground) + np.matrix(center_ground) #[3, 8]

    if g2c_trans.shape[0] == 4: #world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  #only consider the rotation
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground #[3, 8]

    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)
    if True in np.isnan(corners_2d_all):
        print("Invalid projection")
        return None

    corners_2d = corners_2d[0:2].T.tolist()
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])
    return corners_2d



def show_box_with_roll(name_list, thresh=0.5, projectMethod='Ground'):
    """show 2d box and 3d box
        yexiaoqing modified
        # 'Ground': using the ground to camera transformation (denorm: the ground plane equation)
        # 'World': using the extrinsics (world to camera transformation)
    """
    image_root = config.image_dir
    label_dir = config.label_dir
    extrinsics_dir = config.extrinsics_dir
    cal_dir = config.cal_dir
    thresh = config.thresh
    out_dir = config.out_box_dir
    denorm_dir = config.denorm_dir
    use_denorm = False
    use_extrinsic = False
    if projectMethod == 'Ground':
        use_denorm = True
    elif projectMethod == 'World':
        use_extrinsic = True
    
    for i, name in enumerate(name_list):
        img_path = os.path.join(image_root, name)
        name = name.strip()
        name = name.split('/')
        name = name[-1].split('.')[0]
        progress(i, len(name_list))
        detection_file = os.path.join(label_dir, '%s.txt' % (name))
        result = load_detect_data(detection_file)
        if use_denorm:
            denorm_file = os.path.join(denorm_dir, '%s.txt' % (name))
            de_norm = load_denorm_data(denorm_file)  #[ax+by+cz+d=0]
            c2g_trans = compute_c2g_trans(de_norm)
        if use_extrinsic:
            extrinsic_file = os.path.join(extrinsics_dir, '%s.yaml' % (name))
            world2camera = read_kitti_ext(extrinsic_file).reshape((4, 4))
            camera2world = np.linalg.inv(world2camera).reshape(4, 4)
        
        calfile = os.path.join(cal_dir, '%s.txt' % (name))
        p2 = read_kitti_cal(calfile)
        img = cv2.imread(img_path)
        h, w, c = img.shape

        for result_index in range(len(result)):
            t = result[result_index]
            if t.score < thresh:
                continue
            if t.obj_type not in config.color_list.keys():
                continue
            color_type = config.color_list[t.obj_type]
            cv2.rectangle(img, (t.x1, t.y1), (t.x2, t.y2),
                          (255, 255, 255), 1)
            if t.w <= 0.05 and t.l <= 0.05 and t.h <= 0.05: #invalid annotation
                continue

            cam_bottom_center = [t.X, t.Y, t.Z]  # bottom center in Camera coordinate
            if use_extrinsic:
                bottom_center_in_world = camera2world * np.matrix(cam_bottom_center + [1.0]).T
                verts3d = project_3d_world(p2, bottom_center_in_world, t.w, t.h, t.l, t.yaw, camera2world) 
            if use_denorm:
                verts3d = project_3d_ground(p2, np.array(cam_bottom_center), t.w, t.h, t.l, t.yaw, de_norm, c2g_trans)

            if verts3d is None:
                continue
            verts3d = verts3d.astype(np.int32)

            # draw projection
            cv2.line(img, tuple(verts3d[2]), tuple(verts3d[1]), color_type, 2)
            cv2.line(img, tuple(verts3d[1]), tuple(verts3d[0]), color_type, 2)
            cv2.line(img, tuple(verts3d[0]), tuple(verts3d[3]), color_type, 2)
            cv2.line(img, tuple(verts3d[2]), tuple(verts3d[3]), color_type, 2)
            cv2.line(img, tuple(verts3d[7]), tuple(verts3d[4]), color_type, 2)
            cv2.line(img, tuple(verts3d[4]), tuple(verts3d[5]), color_type, 2)
            cv2.line(img, tuple(verts3d[5]), tuple(verts3d[6]), color_type, 2)
            cv2.line(img, tuple(verts3d[6]), tuple(verts3d[7]), color_type, 2)
            cv2.line(img, tuple(verts3d[7]), tuple(verts3d[3]), color_type, 2)
            cv2.line(img, tuple(verts3d[1]), tuple(verts3d[5]), color_type, 2)
            cv2.line(img, tuple(verts3d[0]), tuple(verts3d[4]), color_type, 2)
            cv2.line(img, tuple(verts3d[2]), tuple(verts3d[6]), color_type, 2)
        cv2.imwrite('%s/%s.jpg' % (out_dir, name), img)


if __name__ == '__main__':
    if config.val_list is None:
        name_list = gb.glob(config.image_dir + "/*")
    else:
        val_part_list  = open(config.val_list).readlines()
        name_list = []
        for name in val_part_list:
            name_list.append(name.split('\n')[0] + '.jpg')
        name_list.sort()

    if not os.path.isdir(config.out_box_dir):
      os.makedirs(config.out_box_dir)
   
    #-----------------------------------------------------------------------------------------------------
    # --------------Two approaches can be adopted for projection and visualization------------------------
    # 'Ground': using the ground to camera transformation (denorm: the ground plane equation), default
    # 'World': using the extrinsics (world to camera transformation)
    # show_box_with_roll(name_list, projectMethod='Ground')
    show_box_with_roll(name_list, projectMethod='World')
