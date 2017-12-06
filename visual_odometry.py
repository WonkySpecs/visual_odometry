import cv2
import math
import numpy as np

from helper_functions import *
from gyro import *

HESSIAN_THRESH = 2500
MAX_FEATURES = 300
MIN_FEATURES = 120

#We cut off images below this point, as this contains only the bonnet of the car.
#Reflections in the bonnet can cause problems with feature tracking
IMAGE_Y_CUT = 430

feature_params = dict( maxCorners = MAX_FEATURES,
                       qualityLevel = 0.2,
                       minDistance = 10,
                       blockSize = 10)

class Camera:
	def __init__(self, focal):
		self.focal = focal

class VisualOdometry:
	def __init__(self, frame1, frame2, initial_R, cam):
		self.count = 0
		self.frame_skip_count = 0

		self.cam = cam

		self.total_t = None
		self.total_R = initial_R

		self.noskips_total_t = None
		self.noskips_total_R = None

		self.imu_total_t = None
		self.imu_total_R = None

		self.cur_frame = frame1[:IMAGE_Y_CUT, :]
		
		self.cur_kp = cv2.goodFeaturesToTrack(cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)

		self.update(frame2, None)

	def update(self, new_frame, angles):
		self.prev_frame = self.cur_frame
		self.prev_kp = self.cur_kp
		self.cur_frame = new_frame[:IMAGE_Y_CUT, :]

		if len(self.prev_kp) < MIN_FEATURES:
			self.prev_kp = cv2.goodFeaturesToTrack(cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)

		kp, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, self.cur_frame, self.prev_kp, None)#, **lk_params)
		st = st.reshape(st.shape[0])
		self.cur_kp = kp[st == 1]
		self.prev_kp = self.prev_kp[st == 1]
		
		#Taken from findHomograph() function in mosaic_support, acknowledgements in helper_functions.py
		e_mat, mask = cv2.findEssentialMat(self.prev_kp, self.cur_kp, method = cv2.RANSAC, prob = 0.99)
		
		inliers, R, t, mask = cv2.recoverPose(e_mat, self.prev_kp, self.cur_kp)

		# except:
		# 	print("recoverPose failed")
		# 	print(e_mat)
		# 	exit()

		if self.total_t == None:
			self.total_t = t
			self.noskips_total_t = self.total_t
			self.noskips_total_R = self.total_R
			self.imu_total_t = self.total_t
			self.imu_total_R = self.total_R
		else:
			self.count += 1
			self.noskips_total_t = self.noskips_total_t + (np.matmul(self.noskips_total_R, t))
			self.noskips_total_R = np.matmul(R, self.noskips_total_R)

			if (math.fabs(t[0]) > math.fabs(t[1]) and math.fabs(t[0]) > math.fabs(t[2])):# and (inliers/len(st)) > 0.65:
				self.total_t = self.total_t + (np.matmul(self.total_R, t))
				self.total_R = np.matmul(R, self.total_R)

				self.imu_total_R = angles_to_R(angles[0], angles[1], angles[2])
				self.imu_total_t = self.imu_total_t + (np.matmul(self.imu_total_R, t))
			else:
				self.frame_skip_count += 1

			