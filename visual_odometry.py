import cv2
import math
import numpy as np

from helper_functions import *

HESSIAN_THRESH = 5000

class VisualOdometry:
	def __init__(self, frame1, frame2):
		self.total_t = None
		self.total_R = None
		self.cur_frame = frame1

		self.cur_kp, self.cur_des = getFeatures(self.cur_frame, HESSIAN_THRESH)		


		self.update(frame2)

	def update(self, new_frame):
		self.prev_frame = self.cur_frame
		self.prev_kp, self.prev_des = self.cur_kp, self.cur_des
		self.cur_frame = new_frame

		self.cur_kp, self.cur_des = getFeatures(self.cur_frame, HESSIAN_THRESH)

		good_matches = matchFeatures(self.prev_des, self.cur_des, 50, 0.7)
		#Taken from findHomograph() function in mosaic_support, acknowledgements in helper_functions.py
		pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
		pts2 = np.float32([self.cur_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
		e_mat, mask = cv2.findEssentialMat(pts1, pts2)
		try:
			inliers, R, t, mask = cv2.recoverPose(e_mat, pts1, pts2)
		except:
			print("recoverPose failed")
			print(e_mat)
			exit()

		if self.total_t == None or self.total_R == None:
			self.total_t = t
			self.total_R = R
			print(t)
			print(R)
		else:
			if math.fabs(t[0]) > math.fabs(t[1]) and math.fabs(t[0]) > math.fabs(t[2]):
				self.total_t = self.total_t + (np.matmul(self.total_R, t))
				self.total_R = np.matmul(R, self.total_R)
			else:
				print("Frame where forward is not primary movement ignored")
