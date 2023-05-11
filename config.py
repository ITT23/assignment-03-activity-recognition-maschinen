'''
This module stores defaults values for application
'''
from enum import Enum

DATAPATH = 'data/'


class ActivityType(Enum):
    WAVING = 0
    SHAKING = 1
    LYING = 2


# DIPPID Values
PORT = 5700
