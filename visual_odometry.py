from common_modules import *
from helper_functions import *
from gyro import *

HESSIAN_THRESH = 2500
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

		self.detector = "good"
		self.cam = cam

		self.total_t = None
		self.total_R = initial_R#np.eye(3, 3)

		self.noskips_total_t = None
		self.noskips_total_R = None
		self.imu_total_t = None
		self.imu_total_R = None

		self.cur_cloud = None
		self.prev_cloud = None

		self.cur_frame = frame1[:IMAGE_Y_CUT, :]
		
		self.cur_kp = self.detect_points(self.cur_frame)

		self.average_feature_change = 0

		self.update(frame2, None)

	def update(self, new_frame, angles):
		self.prev_frame = self.cur_frame
		self.prev_kp = self.cur_kp

		#Remove part of image that contains car bonnet
		self.cur_frame = new_frame[:IMAGE_Y_CUT, :]

		#This updates self.prev_kp and self.cur_kp to be matched arrays of good features
		self.update_features()
		
		#Use the matched features to find the essential matrix, decompose to R and t, and compose with total R and t
		self.calc_total_t_and_R(angles)

	def update_features(self):
		if len(self.prev_kp) < MIN_FEATURES:
			self.prev_kp = self.detect_points(self.prev_frame)

		self.average_feature_change = self.feature_correspondences()

	def detect_points(self, img):
		if self.detector == "FAST":
			fast = cv2.FastFeatureDetector_create(15)
			pts =  fast.detect(img, None)
			ptarray = np.array([np.array([feature.pt[0], feature.pt[1]]) for feature in pts], np.float32)
			ptarray = ptarray.reshape((len(ptarray), 1, 2))
			
			return ptarray
		elif self.detector == "good":
			feature_params = dict( maxCorners = 2000,
			                       qualityLevel = 0.01,
			                       minDistance = 7,
			                       blockSize = 7)
			return cv2.goodFeaturesToTrack(cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)
			
		elif self.detector == "SURF":
			surf = cv2.xfeatures2d.SURF_create(20);
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
		dists = abs(self.cur_kp - self.prev_kp).reshape(-1, 2).max(-1)
		return np.median(dists)

	def triangulate_points(self, R, t):
		"""Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""

		# The canonical matrix (set as the origin)
		P0 = np.array([[1, 0, 0, 0],
		               [0, 1, 0, 0],
		               [0, 0, 1, 0]])
		P0 = self.cam.intrinsic_mat.dot(P0)
		# Rotated and translated using P0 as the reference point
		P1 = np.hstack((R, t))
		P1 = self.cam.intrinsic_mat.dot(P1)
		# Reshaped the point correspondence arrays to cv2.triangulatePoints's format
		point1 = self.prev_kp.reshape(2, -1)
		point2 = self.cur_kp.reshape(2, -1)

		return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]

	def calc_scale(self, R, t):
		if self.cur_cloud is not None:
			self.prev_cloud = self.cur_cloud
			self.cur_cloud = self.triangulate_points(R, t)

			scale = 1.0

			min_idx = min([self.cur_cloud.shape[0], self.prev_cloud.shape[0]])
			ratios = []  # List to obtain all the ratios of the distances
			for i in range(1, min_idx):
				Xk = self.cur_cloud[i]
				p_Xk = self.cur_cloud[i - 1]
				Xk_1 = self.prev_cloud[i]
				p_Xk_1 = self.prev_cloud[i - 1]

				if np.linalg.norm(p_Xk - Xk) != 0:
				    ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))
			return np.median(ratios)
		else:
			self.cur_cloud = self.triangulate_points(R, t)
			return 1.0

	def calc_total_t_and_R(self, angles):
		e_mat, mask = cv2.findEssentialMat(self.prev_kp, self.cur_kp, self.cam.intrinsic_mat, method = cv2.RANSAC, prob = 0.999)

		inliers, R, t, mask = cv2.recoverPose(e_mat, self.prev_kp, self.cur_kp, self.cam.intrinsic_mat)

		#The value for t returned is normalised, as it is impossible to calculate scale from the essential matrix
		#Using the approach taken from https://github.com/Transportation-Inspection/visual_odometry,
		#we triangulate cur_kp and prev_kp in 3d space to calculate the scale of the translation
		scale = self.calc_scale(R, t)

		#The t given by recoverPose is (almost) always the opposite of what we want
		if t[2] < 0:
			t = -t

		if self.total_t is None:
			self.total_t = self.total_R.dot(t)
			self.noskips_total_t = self.total_t
			self.noskips_total_R = self.total_R
			self.imu_total_t = self.total_t
			self.imu_total_R = self.total_R
		else:
			self.count += 1
			self.noskips_total_t = self.noskips_total_t + scale * self.noskips_total_R.dot(t)
			self.noskips_total_R = R.dot(self.noskips_total_R)

			#Accept if forwards is the dominant motion, and features have moved 'far enough' (otherwise the car is not moving)
			if (t[2] > math.fabs(t[1]) and t[2] > math.fabs(t[0]) and self.average_feature_change > 2):
				self.total_t = self.total_t + scale * self.total_R.dot(t)
				self.total_R = R.dot(self.total_R)

				self.imu_total_t = self.imu_total_t + scale * self.imu_total_R.dot(t)
				self.imu_total_R = angles_to_R(angles[0], angles[1], angles[2])
			else:
				self.frame_skip_count += 1
				#print(self.frame_skip_count)