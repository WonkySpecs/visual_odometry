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