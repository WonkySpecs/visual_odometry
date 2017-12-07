# getFeatures() and matchFeatures() functions are taken from the mosaic_support
# computeEssentialMat() is adapted from computeHomography() in the same file
# file used for lab work
# Acknowledgements:
# Toby Breckon, toby.breckon@durham.ac.uk
#  bmhr46@durham.ac.uk (2016/17);
# Marc Pare, code taken from:
# https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py

from common_modules import *
import csv

def read_gps_imu_data(dataset_path):
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

    return gps_data, imu_data

#Convert from lon,lat to x, y coordinate spaceused in output trajectory
def convert_gps_to_coords(lon, lat):
    return (440 * lon + 695.01) * 100, (-730 * lat + 39988.18) * 100

#Gives the distance between latitude' longitude points in kilometres
#https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def haversine_dist(lat1, lon1, lat2, lon2):
    lat1    = math.radians(lat1)
    lon1   = math.radians(lon1)
    lat2    = math.radians(lat2)
    lon2   = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return 6371 * c

def get_start_conditions(start_frame):
    if start_frame == 0:
        return 420, 818, None
    elif start_frame == 410:
        return 258, 665, np.array([[-0.75, -0.25, 0.85], [0.2, -0.95, -0.2], [0.75, -0.16, 0.45]])
    elif start_frame == 840:
        return 325, 220, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    elif start_frame == 1800:
        return 180, 262, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    else:
        print("get_start_conditions called for frame where conditions not specified")
        return 400, 400, None