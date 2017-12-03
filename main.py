import cv2
import numpy as np
import math
import os
import csv
import time

from visual_odometry import VisualOdometry
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
vo = None

for frame_id, filename in enumerate(os.listdir(os.path.join(dataset_path, image_path))):
	if filename.endswith(".png"):
		full_file_path = os.path.join(dataset_path, image_path, filename)
		cur_frame = cv2.imread(full_file_path, cv2.IMREAD_COLOR)

		if frame_id == 0:
			first_frame = cur_frame
		elif frame_id == 1:
			second_frame = cur_frame
		else:
			if not vo:
				vo = VisualOdometry(first_frame, second_frame)

			# cur_gps_data = gps_data[frame]
			# cur_imu_data = imu_data[frame]
			# if prev_gps_data:
			# 	gps_dist = haversine_dist(cur_gps_data[1], cur_gps_data[2], prev_gps_data[1], prev_gps_data[2])
			# 	gps_dist *= 0.8
				
			# 	heading = 140 - gyro_to_heading(cur_imu_data[1], cur_imu_data[2], cur_imu_data[3], cur_imu_data[4])
			# 	heading = math.radians(heading)
				
			# 	ground_x += gps_dist * math.sin(heading)
			# 	ground_y += gps_dist * math.cos(heading)
			vo.update(cur_frame)

			tt = vo.total_t

			#print(tt)

			x = tt[0] + X_OFFSET
			y = tt[1] + Y_OFFSET

			cv2.rectangle(output_traj, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)

			#cv2.rectangle(output_traj, (int(ground_x) - 1, int(ground_y) - 1), (int(ground_x) + 1, int(ground_y) + 1), (0, 255, 0), -1)

			cv2.imshow(input_image_windowname, cur_frame)
			final_out = output_traj.copy()
			cv2.putText(final_out, "x:{: 8.3f}    y:{: 8.3f}    z:{: 8.3f}".format(tt[0][0], tt[1][0], tt[2][0]), (10, 22), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 255), 2)
			
			cv2.imshow(output_track_windowname, final_out)
	key = cv2.waitKey(40) & 0xFF
	if key == ord('q'):
		break
cv2.destroyAllWindows()