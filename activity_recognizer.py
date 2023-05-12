'''
This module recognizes activities# this program recognizes activities
'''

from typing import Dict
import config
from scipy import signal
import numpy as np


class Recognizer:
    def __init__(self, classifier):
        self.classifier = classifier
        self.data_list = []
        self.last_row = {}
        self.kernel = signal.gaussian(10, 3)
        self.kernel /= np.sum(self.kernel)

    def process_data(self, acc: Dict[str, Dict[str, float]], gyr: Dict[str, Dict[str, float]], grav: Dict[str, Dict[str, float]]):
        dict_row = {}
        dict_row['acc_x'] = acc['x']
        dict_row['acc_y'] = acc['y']
        dict_row['acc_z'] = acc['z']

        dict_row['gyr_x'] = gyr['x']
        dict_row['gyr_y'] = gyr['y']
        dict_row['gyr_z'] = gyr['z']

        dict_row['grav_x'] = grav['x']
        dict_row['grav_y'] = grav['y']
        dict_row['grav_z'] = grav['z']

        if not self.last_row or self.last_row != dict_row:
            self.last_row = dict_row.copy()
            self.data_list.append(dict_row.copy())

    def predict(self, data):
        parameters = []
        for sensor in config.SENSOR_NAMES:
            data[sensor] = np.convolve(data[sensor], self.kernel, 'same')
            spectrum = np.abs(np.fft.fft(data[sensor]))
            frequencies = np.fft.fftfreq(len(data[sensor]), 1 / config.SAMPLING_RATE)
            mask = frequencies >= 0
            frequency = np.argmax(spectrum[mask] * config.SAMPLING_RATE) / config.SAMPLING_LENGTH_INPUT
            parameters.append(frequency)

        return self.classifier.predict([parameters])