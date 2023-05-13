'''
This module recognizes activities
'''

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
        '''
        Transforms data input to Dict and appends it to data list
        '''
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
        '''
        calculate frequency for every sensor
        sum up frequencies and use this for prediction
        :return: predicted label (0, 1, 2) for given sensor data
        '''
        parameters = []
        data = pd.DataFrame(self.data_list)

        #sum_sensor_amplitudes = 0
        #sensor_amplitudes = []
        #sum_sensor_frequencies = 0
        for sensor in config.SENSOR_NAMES:
            if sensor in data:
                try:
                    print(data[sensor])
                    data[sensor] = np.convolve(data[sensor], self.kernel, 'same')
                    spectrum = np.abs(np.fft.fft(data[sensor]))

                    #amplitudes = 2 / (config.SAMPLING_LENGTH_INPUT * config.SAMPLING_RATE) * np.abs(spectrum)
                    #amplitude = np.mean(amplitudes)


                    frequencies = np.fft.fftfreq(len(data[sensor]), 1 / config.SAMPLING_RATE)
                    mask = frequencies >= 0
                    frequency = np.argmax(spectrum[mask] * config.SAMPLING_RATE) / config.SAMPLING_LENGTH_INPUT

                    #sum_sensor_amplitudes += amplitude
                    #sensor_amplitudes.append(amplitude)
                    #sum_sensor_frequencies += frequency

                    parameters.append(frequency)
                    print(frequency)
                    #parameters.append(amplitude)
                except:
                    print("WARNING: Check if your DIPPID device is still sending data!")
                    continue
        self.data_list.clear()
        sum_sensors = sum(parameters)
        #mean_sensor_amplitudes = sum(sensor_amplitudes)/len(sensor_amplitudes)
        #print(sum_sensor_frequencies, mean_sensor_amplitudes)
        #return self.classifier.predict([[sum_sensor_frequencies, mean_sensor_amplitudes]])
        #return self.classifier.predict([parameters])
        print(sum_sensors)
        return self.classifier.predict([[sum_sensors]])
