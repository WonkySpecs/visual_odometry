import cv2
import numpy as np
import math
import os
import csv
import time

from helper_functions import *

dataset_path = os.path.join(os.curdir, "TTBB-durham-02-10-17-sub5")
image_path = "left-images"

input_image_windowname = "Input image"
output_track_windowname = "Trajectory"
cv2.namedWindow(input_image_windowname, cv2.WINDOW_NORMAL)
cv2.namedWindow(output_track_windowname, cv2.WINDOW_NORMAL)

height = 1000
width = 1000
output_traj = np.zeros((height, width, 3), np.uint8)
x = width / 2
y = height / 2

for filename in os.listdir(os.path.join(dataset_path, image_path)):
	if filename.endswith(".png"):
		print(filename)
		full_file_path = os.path.join(dataset_path, image_path, filename)
		cur_img = cv2.imread(full_file_path, cv2.IMREAD_COLOR)

		try:
			cur_kp, cur_des = getFeatures(cur_img, 10000)

			good_matches = matchFeatures(prev_des, cur_des, 50, 0.7)

			#Taken from findHomograph() function in mosaic_support, acknowledgements in helper_functions.py
			pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
			pts2 = np.float32([cur_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

			e_mat, mask = cv2.findEssentialMat(pts1, pts2)

			inliers, R, t, mask = cv2.recoverPose(e_mat, pts1, pts2)

			try:
				total_t = total_t + (np.matmul(total_R, t))
				total_R = np.matmul(R, total_R)
			except NameError:
				total_t = t
				total_R = R
				continue

			prev_img = cur_img
			prev_kp, prev_des = cur_kp, cur_des

			x = total_t[0] + width / 2
			y = total_t[1] + height / 2

			cv2.circle(output_traj, (x, y), 2, (0, 0, 255))

			#cv2.drawKeypoints(cur_img, cur_kp, cur_img)
			cv2.imshow(input_image_windowname, cur_img)
			cv2.imshow(output_track_windowname, output_traj)

		#Skip first frame, as no previous frame is known
		except NameError:
			prev_img = cur_img
			prev_kp, prev_des = getFeatures(prev_img, 10000)
			continue		

	key = cv2.waitKey(40) & 0xFF
	if key == ord('q'):
		break
cv2.destroyAllWindows()