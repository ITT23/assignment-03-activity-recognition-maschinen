'''
This module stores defaults values for application
'''
from enum import Enum
import pyglet


DATAPATH = 'data/'


class ActivityType(Enum):
    WAVING = 0
    SHAKING = 1
    LYING = 2


# DIPPID Values
PORT = 5700

# for recognition
SAMPLING_RATE_INPUT = pyglet.clock.get_default()
SAMPLING_LENGTH_INPUT = 1

SAMPLING_RATE = 100
SENSOR_NAMES = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']
