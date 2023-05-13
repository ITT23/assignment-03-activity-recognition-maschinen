'''
This module stores defaults values for application
'''
from enum import Enum


DATAPATH = 'data/'
SENSOR_NAMES = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']

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

# approximate frame rate of pyglet window is used as sampling rate for activity recognition
# since the program reads sensor data on every update()-call
SAMPLING_RATE_INPUT = 60

# initiate prediction every 1.5 seconds
SAMPLING_LENGTH_INPUT = 1.5




