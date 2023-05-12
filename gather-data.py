'''
This module gathers sensor data and saves it to /data/FILENAME.csv
'''
from DIPPID import SensorUDP
from typing import Dict
import config
from datetime import datetime
import csv
import time

is_gathering_data = False
sensor = SensorUDP(config.PORT)
label = 0
data_list = []
last_row = {}


def save_data(data_list: list):
    '''
    Transform data list to csv file
    '''
    timestamp = datetime.now().strftime('%d-%m-%y %H-%M-%S')
    fieldnames = list(data_list[0].keys())
    with open(fr'{config.DATAPATH}{label.name} {timestamp}.csv', 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(f=csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)


def process_data(acc: Dict[str, Dict[str, float]], gyr: Dict[str, Dict[str, float]], grav: Dict[str, Dict[str, float]]):
    '''
    Transforms data input to Dict and appends it to data list
    '''
    global last_row

    timestamp = time.time()
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

    if not last_row or last_row != dict_row:
        last_row = dict_row.copy()
        dict_row['label'] = label.value
        dict_row['timestamp'] = timestamp
        data_list.append(dict_row.copy())
        dict_row.clear()


if __name__ == '__main__':
    while True:
        if sensor.get_capabilities:
            if sensor.get_value('button_1') and not is_gathering_data:
                print("Start recording...\nGathering waving data")
                is_gathering_data = True
                label = config.ActivityType.WAVING
                time.sleep(1)
            elif sensor.get_value('button_2') and not is_gathering_data:
                print("Start recording...\nGathering shaking data")
                is_gathering_data = True
                label = config.ActivityType.SHAKING
                time.sleep(1)
            elif sensor.get_value('button_3') and not is_gathering_data:
                print("Start recording...\nGathering lying data")
                is_gathering_data = True
                label = config.ActivityType.LYING
                time.sleep(1)
            elif is_gathering_data:
                if sensor.get_value('button_1') or sensor.get_value('button_2') or sensor.get_value('button_3'):
                    print("Recording stopped.")
                    is_gathering_data = False
                    save_data(data_list)
                    data_list.clear()
                    last_row.clear()
                    time.sleep(1)
                else:
                    accelerometer_data = sensor.get_value('accelerometer')
                    gyroscope_data = sensor.get_value('gyroscope')
                    gravity_data = sensor.get_value('gravity')
                    process_data(accelerometer_data, gyroscope_data, gravity_data)
