import cv2
import math
import numpy as np
import time

from helper_functions import *
from gyro import *

HESSIAN_THRESH = 2500
MAX_FEATURES = 2000
MIN_FEATURES = 1500

#We cut off images below this point, as this contains only the bonnet of the car.
#Reflections in the bonnet can cause problems with feature tracking
IMAGE_Y_CUT = 400

class Camera:
	def __init__(self):
		self.intrinsic_mat = np.array([[399.9745178222656, 0.0, 474.5],
 										[0.0, 399.9745178222656, 262.0],
 										[0.0, 0.0, 1.0]])
		self.focal = self.intrinsic_mat[0, 0]

class VisualOdometry:
	def __init__(self, frame1, frame2, initial_R, cam):
		self.count = 0
		self.frame_skip_count = 0

		self.detector = "FAST"
		self.cam = cam

		self.total_t = None
		self.total_R = initial_R

		self.noskips_total_t = None
		self.noskips_total_R = None

		self.imu_total_t = None
		self.imu_total_R = None

		self.cur_frame = frame1[:IMAGE_Y_CUT, :]
		
		self.cur_kp = self.detect_points(self.cur_frame)

		self.update(frame2, None)

	def update(self, new_frame, angles):
		self.prev_frame = self.cur_frame
		self.prev_kp = self.cur_kp

		#Remove part of image that contains car bonnet
		self.cur_frame = new_frame[:IMAGE_Y_CUT, :]

		#This updates self.prev_kp and self.cur_kp to be matched arrays of features which can be used to calculate the essential matrix
		self.update_features()
		
		self.calc_total_t_and_R(angles)

	def update_features(self):
		if len(self.prev_kp) < MIN_FEATURES:
			self.prev_kp = self.detect_points(self.prev_frame)

		self.feature_correspondences()

	def detect_points(self, img):
		if self.detector == "FAST":
			fast = cv2.FastFeatureDetector_create(30)
			pts =  fast.detect(img, None)
			ptarray = np.array([np.array([feature.pt[0], feature.pt[1]]) for feature in pts], np.float32)
			ptarray = ptarray.reshape((len(ptarray), 1, 2))
			
			return ptarray
		elif self.detector == "good":
			feature_params = dict( maxCorners = MAX_FEATURES,
			                       qualityLevel = 0.2,
			                       minDistance = 10,
			                       blockSize = 10)
			return cv2.goodFeaturesToTrack(cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)
			
		elif self.detector == "SURF":
			surf = cv2.xfeatures2d.SURF_create(40);
			pts, _ = surf.detectAndCompute(img,None);
			ptarray = np.array([np.array([feature.pt[0], feature.pt[1]]) for feature in pts], np.float32)
			ptarray = ptarray.reshape((len(ptarray), 1, 2))
			return ptarray

	def feature_correspondences(self):
		#Using Kanade-Lucas tracking to find flows from points in one frame to the next.
		#Backtracking idea taken from https://github.com/Transportation-Inspection/visual_odometry
		#After finding flows from prev_frame to cur_frame (giving us kp1), we then try to backtrack the flows from cur_frame to prev_frame (giving us kp2).
		#For each point in kp2, we check that it is close to the original point in prev_frame.
		#These close matches are used for computing the essential matrix
		
		dist_threshold = 1
		#lk_params = dict( winSize = (15, 15))
		kp1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, self.cur_frame, self.prev_kp, None)#, **lk_params)
		kp2, st, err = cv2.calcOpticalFlowPyrLK(self.cur_frame, self.prev_frame, kp1, None)#, **lk_params)

		dists = abs(self.prev_kp - kp2).max(-1)
		close = dists < dist_threshold

		cur_kp, prev_kp = [], []

		for i, close_flag in enumerate(close):
			if close_flag:
				cur_kp.append(kp1[i])
				prev_kp.append(kp2[i])

		self.cur_kp = np.array(cur_kp).reshape(-1, 1, 2)
		self.prev_kp = np.array(prev_kp).reshape(-1, 1, 2)

	def calc_total_t_and_R(self, angles):
		#Taken from findHomography() function in mosaic_support, acknowledgements in helper_functions.py
		e_mat, mask = cv2.findEssentialMat(self.prev_kp, self.cur_kp, self.cam.intrinsic_mat)#, method = cv2.RANSAC, prob = 0.999)

		inliers, R, t, mask = cv2.recoverPose(e_mat, self.prev_kp, self.cur_kp)

		#Band aid solution to recoverPose being dodgy
		if t[2] < 0:
			R1, R2, test = cv2.decomposeEssentialMat(e_mat)
			print("wtf")
			t = -t

		if self.total_t == None:
			self.total_t = self.total_R.dot(t)
			self.noskips_total_t = self.total_t
			self.noskips_total_R = self.total_R
			self.imu_total_t = self.total_t
			self.imu_total_R = self.total_R
		else:
			self.count += 1
			self.noskips_total_t = self.noskips_total_t + self.noskips_total_R.dot(t)
			self.noskips_total_R = R.dot(self.noskips_total_R)

			#Accept if moving forwards more than noise could produce, forwards is the dominant motion, and inliers is high enough
			if t[2] > 0.4 and t[2] > math.fabs(t[1]) and t[2] > math.fabs(t[0]) and ((inliers/len(self.cur_kp)) > 0.6):
				self.total_t = self.total_t + self.total_R.dot(t)
				self.total_R = R.dot(self.total_R)

				self.imu_total_R = angles_to_R(angles[0], angles[1], angles[2])
				self.imu_total_t = self.imu_total_t + self.imu_total_R.dot(t)
			else:
				self.frame_skip_count += 1
				print(self.frame_skip_count)