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

# Takes an image and a Hessian threshold value and
# returns the SURF features points (kp) and descriptors (des) of image
# (for SURF features - Hessian threshold of typically 400-1000 can be used)

def getFeatures(img, thres):
    surf = cv2.xfeatures2d.SURF_create(thres)
    kp, des = surf.detectAndCompute(img,None)
    return kp, des

#####################################################################

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

def computeEssentialMat(kp1, kp2, good_matches):
    #compute the transformation
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2);
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2);
    return cv2.findEssentialMat(pts1, pts2)

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

def convert_gps_to_coords(lon, lat):
    # 0: -1.570038 -> 420
    #    54.767093 -> 818

    # 720:
    #     -1.571328 -> 354
    #     54.772757 -> 420

    # 1100:
    #     -1.575188 -> 200
    #     54.776528 -> 135

    return (440 * lon + 695.01) * 100, (-730 * lat + 39988.18) * 100

def IMU_matrices(prev_imu, cur_imu):
    time_passed = cur_imu[0] - prev_imu[0]
    roll, pitch, heading = gyro_to_angles(cur_imu[1], cur_imu[2], cur_imu[3], cur_imu[4])
    R = angles_to_R(roll, pitch, heading)

    t = np.array([cur_imu[5] - prev_imu[5], cur_imu[6] - prev_imu[6], cur_imu[7] - prev_imu[7]])
    t = t.reshape((3, 1))

    return t, R