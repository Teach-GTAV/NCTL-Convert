#!/usr/bin/python
import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np
import utils as utils
__author__ = "maxtom"
__email__  = "hitmaxtom@gmail.com"

class odome:
    """Load and parse odometry benchmark data into a usable format."""

    def __init__(self, base_path, sequence, frame_range=None):
        """Set the path."""
        self.sequence = sequence
        self.sequence_path = os.path.join(base_path, 'sequences', sequence)
        self.pose_path = os.path.join(base_path, 'poses')
        self.frame_range = frame_range

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.sequence_path, 'calib.txt')
        filedata = utils.read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def load_poses(self):
        """Load ground truth poses from file."""
        print('Loading poses for sequence ' + self.sequence + '...')

        pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

        # Read and parse the poses
        try:
            self.T_w_cam0 = []
            with open(pose_file, 'r') as f:
                for line in f.readlines():
                    T = np.fromstring(line, dtype=float, sep=' ')
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    self.T_w_cam0.append(T)

        except FileNotFoundError:
            print('Ground truth poses are not avaialble for sequence ' +
                  self.sequence + '.')

    def load_pose(self, id):
        return self.T_w_cam0[id]

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_path = os.path.join(self.sequence_path, 'velodyne', '*.bin')
        velo_files = sorted(glob.glob(velo_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            velo_files = [velo_files[i] for i in self.frame_range]

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        self.velo_files = velo_files
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        #self.velo = utils.load_velo_scans(velo_files)

    def load_scan(self,id):
        self.vel = utils.load_velo_scan(self.velo_files, id)
        return self.vel
