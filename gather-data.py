'''
This module gathers sensor data and saves it to /data/FILENAME.csv
'''
from DIPPID import SensorUDP
from typing import Dict
import config
from datetime import datetime
import csv

gather_in_progress = False
sensor = SensorUDP(config.PORT)
label = ''
data_list = []

def save_data(data_list: list):
    timestamp = datetime.now().strftime('%d/%m/%y %H-%M-%S')
    fieldnames = list(data_list[0].keys())
    with open(fr'{config.DATAPATH}{timestamp} {label}', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csvfile=csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)

def process_data(acc: Dict[str, Dict[str, float]], gyr: Dict[str, Dict[str, float]], grav: Dict[str, Dict[str, float]]):
    '''
    Creates a Dict which can be used 
    '''
    timestamp = datetime.now()
    dict_row ={'label': label, 'timestamp': timestamp}
    dict_row['acc_x'] = acc[1]['x']
    dict_row['acc_y'] = acc[1]['y']
    dict_row['acc_z'] = acc[1]['z']

    dict_row['gyr_x'] = gyr[1]['x']
    dict_row['gyr_y'] = gyr[1]['y']
    dict_row['gyr_z'] = gyr[1]['z']

    dict_row['grav_x'] = grav[1]['x']
    dict_row['grav_y'] = grav[1]['y']
    dict_row['grav_z'] = grav[1]['z']

    data_list.append(dict_row)

while True:
    if (sensor.has_capability('button_1')):
        if sensor.get_value('button_1')  and not gather_data: # Wie schaut der Value vom Button aus?
            gather_data = True
            label = 'waving'
        elif sensor.get_value('button_2')  and not gather_data:
            gather_data = True
            label = 'shaking'
        elif sensor.get_value('button_2')  and not gather_data:
            gather_data = True
            label = 'standing'
        if gather_data:
            if sensor.get_value('button_1') or sensor.get_value('button_2') or sensor.get_value('button_3'):
                gather_data = False
                save_data(data_list)
                data_list = []
            accelerometer_data = sensor.get_value('accelerometer')
            gyroscope_data = sensor.get_value('gyroscope')
            gravity_data = sensor.get_value('gravity')
            process_data(accelerometer_data, gyroscope_data, gravity_data)
