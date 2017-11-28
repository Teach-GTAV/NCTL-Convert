#!/usr/bin/python

from collections import namedtuple

import matplotlib.image as mpimg
import numpy as np
import struct

import rospy
import tf

from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped

__author__ = "maxtom"
__email__  = "hitmaxtom@gmail.com"

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def load_velo_scans(velo_files):
    """Helper method to parse velodyne binary files into a list of scans."""
    scan_list = []
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        scan_list.append(scan.reshape((-1, 4)))

    return scan_list

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def load_velo_scan(velo_files, id):

    f_bin = open(velo_files[id], "r")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == '': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)
        hits += [[x, y, z]]

    f_bin.close()

    hits = np.asarray(hits)
    scan = hits.reshape((-1, 3))
    return scan

def pose2tf(pose, timestamp, frame_id, child_frame_id):
    tf_msg = tfMessage()
    geo_msg = TransformStamped()
    geo_msg.header.stamp = timestamp
    geo_msg.header.frame_id = frame_id
    geo_msg.child_frame_id = child_frame_id
    geo_msg.transform.translation.x = pose[0]
    geo_msg.transform.translation.y = pose[1]
    geo_msg.transform.translation.z = pose[2]
    
    angles = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5]) 
    geo_msg.transform.rotation.x = angles[0]
    geo_msg.transform.rotation.y = angles[1]
    geo_msg.transform.rotation.z = angles[2]
    geo_msg.transform.rotation.w = angles[3]
    tf_msg.transforms.append(geo_msg)

    return tf_msg
