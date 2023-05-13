'''
This module stores defaults values for application
'''
from enum import Enum


DATAPATH = 'data/'
SENSOR_NAMES = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']
SENSOR_NAMES_W_AMPL = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z', 'acc_x_ampl', 'acc_y_ampl', 'acc_z_ampl', 'gyr_x_ampl', 'gyr_y_ampl', 'gyr_z_ampl', 'grav_x_ampl', 'grav_y_ampl', 'grav_z_ampl']

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 900
BACKGROUND_COLOR_R = 0.61176
BACKGROUND_COLOR_G = 0
BACKGROUND_COLOR_B = 0.29412
BACKGROUND_COLOR_T = 0


class ActivityType(Enum):
    WAVING = 0
    SHAKING = 1
    LYING = 2


# DIPPID Port
PORT = 5700

# approximate sampling rate in train data
SAMPLING_RATE = 100

# filter threshold
THRESHOLD = 0.05

# approximate frame rate of pyglet window is used as sampling rate for activity recognition
# since the program reads sensor data on every update()-call
SAMPLING_RATE_INPUT = 60

# initiate prediction every 1.5 seconds
SAMPLING_LENGTH_INPUT = 3




