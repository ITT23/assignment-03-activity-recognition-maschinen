"""
This module visualizes activity predictions
"""
import os
import pyglet
import config
from typing import Dict
from activity_recognizer import Recognizer

# img source waving: https://creazilla.com/nodes/46453-waving-hand-emoji-clipart
# img source shaking: http://www.onlinewebfonts.com
# img source lying: https://www.clipartmax.com/middle/m2H7d3N4d3m2d3A0_sleeping-stick-figure-clipart-lying-person-icon-png/


class Visualizer:
    def __init__(self, classifier):
        self.recognizer = Recognizer(classifier)
        self.waving_img = pyglet.resource.image(os.path.normpath('assets/waving.png'))
        self.lying_img = pyglet.resource.image(os.path.normpath('assets/lying.png'))
        self.shaking_img = pyglet.resource.image(os.path.normpath('assets/shaking.png'))
        self.visualization_sprite = pyglet.sprite.Sprite(img=self.lying_img)
        self.counter = 0

    def handle_prediction(self, prediction: int):
        """
        show image matching to performed/predicted activity
        :param prediction: predicted label (0, 1, 2)
        """
        prediction = prediction[0]
        if prediction == config.ActivityType.LYING.value:
            print('lying')
            self.visualization_sprite.image = self.lying_img
        elif prediction == config.ActivityType.WAVING.value:
            print('waving')
            self.visualization_sprite.image = self.waving_img
        elif prediction == config.ActivityType.SHAKING.value:
            print('shaking')
            self.visualization_sprite.image = self.shaking_img

    def update(self, acc_data: Dict[str, Dict[str, float]], gyr_data: Dict[str, Dict[str, float]], grav_data: Dict[str, Dict[str, float]]):
        """
        process and save captured sensor data, then
        automatically initiate prediction every <SAMPLING_LENGTH_INPUT> seconds
        :param acc_data: captured accelerometer data
        :param gyr_data: captured gyroscope data
        :param grav_data: captured gravity data
        """
        self.counter += 1
        self.recognizer.process_data(acc_data, gyr_data, grav_data)
        if self.counter >= config.SAMPLING_RATE_INPUT * config.SAMPLING_LENGTH_INPUT:
            self.counter = 0
            prediction = self.recognizer.predict()
            self.handle_prediction(prediction)

    def draw(self):
        self.visualization_sprite.draw()
