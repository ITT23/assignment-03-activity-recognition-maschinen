"""
This module recognizes activities
"""

from typing import Dict
import config
from scipy import signal
import numpy as np
import pandas as pd


class Recognizer:
    def __init__(self, classifier):
        self.classifier = classifier
        self.data_list = []
        self.last_row = {}
        self.kernel = signal.gaussian(10, 3)
        self.kernel /= np.sum(self.kernel)

    def process_data(self, acc: Dict[str, Dict[str, float]], gyr: Dict[str, Dict[str, float]], grav: Dict[str, Dict[str, float]]):
        """
        Transforms data input to Dict and appends it to data list
        :param acc: captured accelerometer data
        :param gyr: captured gyroscope data
        :param grav: captured gravity data
        """
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

        # only use data if it changed in comparison to last
        if not self.last_row or self.last_row != dict_row:
            self.last_row = dict_row.copy()
            self.data_list.append(dict_row.copy())

    def predict(self):
        """
        calculate amplitude of acc_y and frequency for every sensor
        sum up frequencies and use this as well as the acc_y amplitude for prediction
        :return: predicted label (0, 1, 2) for given sensor data
        """
        data = pd.DataFrame(self.data_list)
        sum_frequencies = 0
        amplitude_acc_y = 0
        if 'acc_y' in data:
            amplitude_acc_y = data['acc_y'].max() - data['acc_y'].min()
        for sensor in config.SENSOR_NAMES:
            if sensor in data:
                try:
                    data[sensor] = np.convolve(data[sensor], self.kernel, 'same')
                    spectrum = np.abs(np.fft.fft(data[sensor]))
                    frequencies = np.fft.fftfreq(len(data[sensor]), 1 / config.SAMPLING_RATE)
                    mask = frequencies >= 0
                    frequency = np.argmax(spectrum[mask] * config.SAMPLING_RATE) / config.SAMPLING_LENGTH_INPUT
                    sum_frequencies += frequency
                except:
                    print("WARNING: Check if your DIPPID device is still sending data!")
                    continue
        self.data_list.clear()
        return self.classifier.predict([[sum_frequencies, amplitude_acc_y]])
