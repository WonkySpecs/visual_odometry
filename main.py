from common_modules import *
import os
import csv

from visual_odometry import VisualOdometry, Camera
from helper_functions import *
from gyro import *

dataset_path = os.path.join(os.curdir, "TTBB-durham-02-10-17-sub5")
image_path = "left-images"

#Get (noisy) ground truth gps and imu data
gps_data, imu_data = read_gps_imu_data(dataset_path)

input_image_windowname = "Input image"
output_track_windowname = "Trajectory"
test_windowname = "Test"

cv2.namedWindow(input_image_windowname, cv2.WINDOW_NORMAL)
cv2.namedWindow(output_track_windowname)
cv2.namedWindow(test_windowname)

# start_frames = [0, 410, 840, 1800]
# num_frames_to_run = 500
start_frames = [0]
num_frames_to_run = 3000

for skip_to in start_frames:
	errors = {"tt":[],
					"tt2":[],
					"gps":[]}
	output_traj = cv2.imread("sat_img.png", cv2.IMREAD_COLOR)
	cv2.rectangle(output_traj, (320, 3), (620, 75), (0, 0, 0), -1)
	vo = None
	cam = Camera()
	X_START, Y_START, initial_R = get_start_conditions(skip_to)
	
	print("Start frame {}:".format(skip_to))

	for frame_id, filename in enumerate(os.listdir(os.path.join(dataset_path, image_path))):
		if frame_id > skip_to + num_frames_to_run:
			break

		if skip_to > frame_id:
			continue

		if filename.endswith(".png"):
			full_file_path = os.path.join(dataset_path, image_path, filename)
			cur_frame = cv2.imread(full_file_path, cv2.IMREAD_COLOR)
			smoothed = cv2.GaussianBlur(cur_frame, (3, 3), 0)
			cur_gps_data = gps_data[frame_id]
			cur_imu_data = imu_data[frame_id]

			if frame_id == skip_to:
				first_frame = smoothed
			elif frame_id == skip_to + 1:
				second_frame = smoothed
			else:
				roll, pitch, heading = gyro_to_angles(cur_imu_data[1], cur_imu_data[2], cur_imu_data[3], cur_imu_data[4])
				angles = (roll, pitch, heading)
				if not vo:
					try:
						vo = VisualOdometry(first_frame, second_frame, initial_R, cam)
					except:
						initial_R = angles_to_R(roll, pitch, heading)
						vo = VisualOdometry(first_frame, second_frame, initial_R, cam)

				vo.update(smoothed, angles, (gps_data[frame_id - 1], cur_gps_data))

				tt = vo.total_t
				tt2 = vo.noskips_total_t
				gps_assist_tt = vo.gps_total_t
				
				x = tt[2] + X_START
				y = Y_START - tt[0]

				x2 = tt2[2] + X_START
				y2 = Y_START - tt2[0]

				gps_assist_x = gps_assist_tt[2] + X_START
				gps_assist_y = Y_START - gps_assist_tt[0]

				gps_x, gps_y = convert_gps_to_coords(cur_gps_data[2], cur_gps_data[1])
				errors["tt"].append(math.sqrt(abs(x - gps_x)**2 + abs(y - gps_y)**2))
				errors["tt2"].append(math.sqrt(abs(x2 - gps_x)**2 + abs(y2 - gps_y)**2))
				errors["gps"].append(math.sqrt(abs(gps_assist_x - gps_x)**2 + abs(gps_assist_y - gps_y)**2))
				gps_x = int(gps_x)
				gps_y = int(gps_y)

				cv2.rectangle(output_traj, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 140), -1)
				cv2.rectangle(output_traj, (x2 - 1, y2 - 1), (x2 + 1, y2 + 1), (0, 140, 0), -1)
				cv2.rectangle(output_traj, (gps_x - 1, gps_y - 1), (gps_x + 1, gps_y + 1), (140, 0, 0), -1)
				cv2.rectangle(output_traj, (gps_assist_x - 1, gps_assist_y - 1), (gps_assist_x + 1, gps_assist_y + 1), (255, 255, 255), -1)
				
				final_out = output_traj.copy()
				cv2.putText(final_out, "Thresh:    x:{: 4.1f}  y:{: 4.1f}  error:{: 4.2f}".format(x[0], y[0], errors["tt"][-1]), (325, 15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
				cv2.putText(final_out, "No thresh: x:{: 4.1f}  y:{: 4.1f}  error:{: 4.2f}".format(x2[0], y2[0], errors["tt2"][-1]), (325, 33), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
				cv2.putText(final_out, "GPS assist:x:{: 4.1f}  y:{: 4.1f}  error:{: 4.2f}".format(gps_assist_x[0], gps_assist_y[0], errors["gps"][-1]), (325, 51), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
				cv2.putText(final_out, "GPS:       x:{: 4.1f}  y:{: 4.1f}  error: -".format(gps_x, gps_y), (325, 69), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
				cv2.rectangle(final_out, (x2 - 1, y2 - 1), (x2 + 1, y2 + 1), (30, 255, 30), -1)
				cv2.rectangle(final_out, (gps_x - 1, gps_y - 1), (gps_x + 1, gps_y + 1), (255, 30, 30), -1)
				cv2.rectangle(final_out, (x - 1, y - 1), (x + 1, y + 1), (30, 30, 255), -1)

				for pair in zip(vo.prev_kp, vo.cur_kp):
					prev_pt = (pair[0][0, 0], pair[0][0, 1])
					cur_pt = (pair[1][0, 0], pair[1][0, 1])
					cur_frame = cv2.arrowedLine(cur_frame, cur_pt, prev_pt, (0, 255, 0))

				cv2.imshow(input_image_windowname, cur_frame)
				cv2.imshow(output_track_windowname, final_out)
				cv2.imshow(test_windowname, vo.cur_frame)

		key = cv2.waitKey(40) & 0xFF
		if key == ord('x'):
			exit()
		elif key == ord('n'):
			break
	for k, v in errors.items():
		print("Average error for {}: {}".format(k, np.mean(errors[k])))
cv2.destroyAllWindows()