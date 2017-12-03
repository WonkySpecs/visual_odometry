import cv2
import numpy as np
import math
import os
import csv
import time

from helper_functions import *
from gyro import gyro_to_angles

dataset_path = os.path.join(os.curdir, "TTBB-durham-02-10-17-sub5")
image_path = "left-images"

#Get (noisy) ground truth gps data
gps_data = []
with open(os.path.join(dataset_path, "GPS.csv")) as GPS_file:
	reader = csv.reader(GPS_file, delimiter=',')
	first_line = True
	for line in reader:
		if not first_line:
			gps_data.append([float(v) for v in line])
		else:
			first_line = False

imu_data = []
with open(os.path.join(dataset_path, "IMU.csv")) as GPS_file:
	reader = csv.reader(GPS_file, delimiter=',')
	first_line = True
	for line in reader:
		if not first_line:
			imu_data.append([float(v) for v in line])
		else:
			col_names = line
			first_line = False

for i in range(5):
	for a, b in zip(col_names, imu_data[i]):
		print("{} : {}".format(a, b))
	print("------\n")

input_image_windowname = "Input image"
output_track_windowname = "Trajectory"

cv2.namedWindow(input_image_windowname, cv2.WINDOW_NORMAL)
cv2.namedWindow(output_track_windowname)

sat_img = cv2.imread("sat_img.png", cv2.IMREAD_COLOR)
output_traj = cv2.resize(sat_img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
cv2.rectangle(output_traj, (3, 3), (430, 30), (0, 0, 0), -1)
X_OFFSET = x = ground_x = 82
Y_OFFSET = y = ground_y = 400

cur_gps_data = prev_gps_data = None

for frame, filename in enumerate(os.listdir(os.path.join(dataset_path, image_path))):
	if filename.endswith(".png"):
		full_file_path = os.path.join(dataset_path, image_path, filename)
		cur_img = cv2.imread(full_file_path, cv2.IMREAD_COLOR)

		try:
			cur_kp, cur_des = getFeatures(cur_img, 10000)

			good_matches = matchFeatures(prev_des, cur_des, 50, 0.7)

			#Taken from findHomograph() function in mosaic_support, acknowledgements in helper_functions.py
			pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
			pts2 = np.float32([cur_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

			e_mat, mask = cv2.findEssentialMat(pts1, pts2)

			try:
				inliers, R, t, mask = cv2.recoverPose(e_mat, pts1, pts2)
			except:
				print("recoverPose failed")
				print(e_mat)
				exit()

			# cur_gps_data = gps_data[frame]
			# cur_imu_data = imu_data[frame]
			# if prev_gps_data:
			# 	gps_dist = haversine_dist(cur_gps_data[1], cur_gps_data[2], prev_gps_data[1], prev_gps_data[2])
			# 	gps_dist *= 0.8
				
			# 	heading = 140 - gyro_to_heading(cur_imu_data[1], cur_imu_data[2], cur_imu_data[3], cur_imu_data[4])
			# 	heading = math.radians(heading)
				
			# 	ground_x += gps_dist * math.sin(heading)
			# 	ground_y += gps_dist * math.cos(heading)

			if math.fabs(t[0]) > math.fabs(t[1]) and math.fabs(t[0]) > math.fabs(t[2]):
				try:
					total_t = total_t + (np.matmul(total_R, t))
					total_R = np.matmul(R, total_R)

					x = total_t[0] + X_OFFSET
					y = total_t[1] + Y_OFFSET

					#cv2.circle(output_traj, (x, y), 2, (0, 0, 255))
					cv2.rectangle(output_traj, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)

					#cv2.rectangle(output_traj, (int(ground_x) - 1, int(ground_y) - 1), (int(ground_x) + 1, int(ground_y) + 1), (0, 255, 0), -1)
				except NameError:
					total_t = np.zeros((3, 1), np.float64)
					total_R = R
					continue
			else:
				print("Frame where forward is not primary movement ignored")

			prev_img = cur_img
			prev_kp, prev_des = cur_kp, cur_des
			prev_gps_data = cur_gps_data

			cv2.imshow(input_image_windowname, cur_img)
			final_out = output_traj.copy()
			cv2.putText(final_out, "x:{: 8.3f}    y:{: 8.3f}    z:{: 8.3f}".format(total_t[0][0], total_t[1][0], total_t[2][0]), (10, 22), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 255), 2)
			
			cv2.imshow(output_track_windowname, final_out)

		#Skip first frame, as no previous frame is known
		except NameError:
			prev_img = cur_img
			prev_kp, prev_des = getFeatures(prev_img, 10000)
			prev_gps_data = cur_gps_data
			continue		

	key = cv2.waitKey(40) & 0xFF
	if key == ord('q'):
		break
cv2.destroyAllWindows()