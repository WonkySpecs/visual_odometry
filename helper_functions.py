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
