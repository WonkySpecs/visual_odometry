# getFeatures() and matchFeatures() functions are taken from the mosaic_support
# computeEssentialMat() is adapted from computeHomography() in the same file
# file used for lab work
# Acknowledgements:
# Toby Breckon, toby.breckon@durham.ac.uk
#  bmhr46@durham.ac.uk (2016/17);
# Marc Pare, code taken from:
# https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py

import numpy as np
import cv2
import math

from gyro import *

# Performs FLANN-based feature matching of descriptor from 2 images
# returns 'good matches' based on their distance
# typically number_of_checks = 50, match_ratio = 0.7

def matchFeatures(des1, des2, number_of_checks, match_ratio):
    index_params = dict(algorithm = 0, trees = 1) #FLANN_INDEX_KDTREE = 0
    search_params = dict(checks = number_of_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)
    #matchesMask = [[0,0] for i in range(len(matches))]
    good_matches = [];
    for i,(m,n) in enumerate(matches):
        if m.distance < match_ratio * n.distance:   #filter out 'bad' matches
            #matchesMask[i]=[1,0];
            good_matches.append(m);
    return good_matches

def haversine_dist(lat1, long1, lat2, long2):
    lat1    = math.radians(lat1)
    long1   = math.radians(long1)
    lat2    = math.radians(lat2)
    long2   = math.radians(long2)

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlong / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return 6371 * c * 1000

#Convert from lon,lat to x, y coordinate spaceused in output trajectory
def convert_gps_to_coords(lon, lat):
    return (440 * lon + 695.01) * 100, (-730 * lat + 39988.18) * 100

def IMU_matrices(prev_imu, cur_imu):
    time_passed = cur_imu[0] - prev_imu[0]
    roll, pitch, heading = gyro_to_angles(cur_imu[1], cur_imu[2], cur_imu[3], cur_imu[4])
    R = angles_to_R(roll, pitch, heading)

    t = np.array([cur_imu[5] - prev_imu[5], cur_imu[6] - prev_imu[6], cur_imu[7] - prev_imu[7]])
    t = t.reshape((3, 1))

    return t, R