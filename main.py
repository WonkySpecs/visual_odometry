import cv2
import numpy as np
import math
import os
import csv
import time

from visual_odometry import VisualOdometry, Camera
from helper_functions import *
from gyro import *

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

input_image_windowname = "Input image"
output_track_windowname = "Trajectory"
test_windowname = "Test"

cv2.namedWindow(input_image_windowname, cv2.WINDOW_NORMAL)
cv2.namedWindow(output_track_windowname)
#cv2.namedWindow(test_windowname)

output_traj = cv2.imread("sat_img.png", cv2.IMREAD_COLOR)
#output_traj = cv2.resize(sat_img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
cv2.rectangle(output_traj, (3, 3), (430, 80), (0, 0, 0), -1)
X_START = x = 420
Y_START = y = 818

cur_gps_data = prev_gps_data = None
vo = None
cam = Camera(4.8)

for frame_id, filename in enumerate(os.listdir(os.path.join(dataset_path, image_path))):
	if filename.endswith(".png"):
		full_file_path = os.path.join(dataset_path, image_path, filename)
		cur_frame = cv2.imread(full_file_path, cv2.IMREAD_COLOR)
		smoothed = cv2.GaussianBlur(cur_frame, (5, 5), 0)
		cur_gps_data = gps_data[frame_id]
		cur_imu_data = imu_data[frame_id]

		if frame_id == 0:
			first_frame = smoothed
		elif frame_id == 1:
			second_frame = smoothed
		else:
			roll, pitch, heading = gyro_to_angles(cur_imu_data[1], cur_imu_data[2], cur_imu_data[3], cur_imu_data[4])
			angles = (roll, pitch, heading)
			if not vo:
				initial_R = angles_to_R(roll, pitch, heading)				
				vo = VisualOdometry(first_frame, second_frame, initial_R, cam)

			vo.update(smoothed, angles)

			tt = vo.total_t
			tt2 = vo.noskips_total_t
			imu_tt = vo.imu_total_t

			x = tt[0] + X_START
			y = tt[1] + Y_START

			x2 = tt2[0] + X_START
			y2 = tt2[1] + Y_START

			imu_x = imu_tt[0] + X_START
			imu_y = imu_tt[1] + Y_START

			cv2.rectangle(output_traj, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 180), -1)
			cv2.rectangle(output_traj, (x2 - 1, y2 - 1), (x2 + 1, y2 + 1), (0, 180, 0), -1)
			cv2.rectangle(output_traj, (imu_x - 1, imu_y - 1), (imu_x + 1, imu_y + 1), (180, 0, 0), -1)
			
			final_out = output_traj.copy()
			#cv2.putText(final_out, "x:{: 8.3f}    y:{: 8.3f}    z:{: 8.3f}".format(tt[0][0], tt[1][0], tt[2][0]), (10, 22), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 255), 2)
			#cv2.putText(final_out, "x:{: 8.3f}    y:{: 8.3f}    z:{: 8.3f}".format(tt2[0][0], tt2[1][0], tt2[2][0]), (10, 46), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 255), 2)
			#cv2.putText(final_out, "x:{: 8.3f}    y:{: 8.3f}    z:{: 8.3f}".format(imu_tt[0][0], tt[1][0], imu_tt[2][0]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 255), 2)
			cv2.putText(final_out, "heading: {}".format(heading), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
			cv2.rectangle(final_out, (x - 1, y - 1), (x + 1, y + 1), (20, 20, 255), -1)
			cv2.rectangle(final_out, (x2 - 1, y2 - 1), (x2 + 1, y2 + 1), (20, 255, 20), -1)
			cv2.rectangle(final_out, (imu_x - 1, imu_y - 1), (imu_x + 1, imu_y + 1), (255, 20, 20), -1)

			for pair in zip(vo.prev_kp, vo.cur_kp):
				prev_pt = (pair[0][0, 0], pair[0][0, 1])
				cur_pt = (pair[1][0, 0], pair[1][0, 1])
				cur_frame = cv2.arrowedLine(cur_frame, cur_pt, prev_pt, (0, 255, 0))

			cv2.imshow(input_image_windowname, cur_frame)
			cv2.imshow(output_track_windowname, final_out)

	key = cv2.waitKey(40) & 0xFF
	if key == ord('q'):
		break
cv2.destroyAllWindows()