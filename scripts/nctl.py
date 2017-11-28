#!/usr/bin/python
import rospy
import tf
import math
import sys

import std_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

import matplotlib.pyplot as plt
import numpy as np
from odometry import odome
from glob import glob
import os, sys
import rosbag, rospy

from utils import pose2tf

__author__ = "maxtom"
__email__ = "hitmaxtom@gmail.com"

def talker():
    rospy.init_node('nctl_pub', anonymous=True)
    pub = rospy.Publisher('/nctl/cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)
    header = std_msgs.msg.Header()
    header.frame_id = 'velodyne'

    # Change this to the directory where you store KITTI data
    basedir = '/data1/NCLT'
    sequence = '2012-01-08'

    velo_path = os.path.join(basedir, 'sequences', sequence, "velodyne_sync/*.bin")
    files = glob(velo_path)
    
    bag_path = os.path.join(basedir, 'bags', sequence+'.bag')

    # bag file
    bag = rosbag.Bag(bag_path, 'w')
    # Optionally, specify the frame range to load
    frame_range = range(5, len(files)-5, 1)
    
    # Load the data
    # dataset = pykitti.odometry(basedir, sequence)
    dataset = odome(basedir, sequence, frame_range)
    np.set_printoptions(precision=4, suppress=True)
    
    # Load some data
    dataset.load_poses()        # Ground truth poses are loaded as 4x4 arrays
    dataset.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]
    
    id=0
    while not rospy.is_shutdown():

        pcd, time_id = dataset.load_scan(id)

        timestamp = rospy.Time.from_sec(np.float64(time_id)/1e6)

        header.stamp = timestamp

        cloud  = pcl2.create_cloud_xyz32(header, pcd)

        pose  = dataset.load_pose(time_id)[1:]

        # Publish Tf
        br = tf.TransformBroadcaster()

        tf_msgs = []

        tf_msg = pose2tf(np.array([0.0, 0.0, 0.0, np.pi, 0.0, 0.0]), timestamp, "world", "cam_world")
        tf_msgs.append(tf_msg)

        tf_msg = pose2tf(pose, timestamp, "cam_world", "base_link")
        tf_msgs.append(tf_msg)

        tf_msg = pose2tf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), timestamp, "base_link", "velodyne",)
        tf_msgs.append(tf_msg)


        br.sendTransform((0.0, 0.0, 0.0),
                         tf.transformations.quaternion_from_euler(np.pi, 0.0, 0.0),
                         timestamp,
                         "cam_world", "world")

        br.sendTransform((pose[0], pose[1], pose[2]),
                         tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5]),
                         timestamp,
                         "base_link", "cam_world")

        br.sendTransform((0.0, 0.0, 0.0),
                         tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0),
                         timestamp,
                         "velodyne", "base_link")


        rospy.loginfo("new data")
        pub.publish(cloud)

        # write bag
        bag.write('/nctl/cloud', cloud, t=timestamp)
        for msg in tf_msgs:
            bag.write('/tf', msg, msg.transforms[0].header.stamp )
    
        id=id+1
        rate.sleep()

    bag.close()
    rospy.loginfo("done loop")

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
