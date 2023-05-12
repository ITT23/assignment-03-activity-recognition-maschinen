import os
import pandas as pd
import config
import numpy as np
from scipy import signal

from sklearn import model_selection, metrics, svm


class Trainer:
    def __init__(self):
        self.classifier = None
        self.train_data = None
        self.test_data = None
        self.train_data_list = []

    def read_data(self):
        print("reading train data...")
        dataframes = []
        for filename in os.listdir(config.DATAPATH):
            file = os.path.join(config.DATAPATH, filename)
            if os.path.isfile(file):
                dataframe = pd.read_csv(file)
                splitted = self.split(dataframe)
                for split in splitted:
                    self.calc_frequencies(split)
        self.train_data = pd.DataFrame(self.train_data_list)
        print(self.train_data)

    def split(self, dataframe):
        cut_df = dataframe.copy()
        cut_df = cut_df[(cut_df['timestamp'] >= (cut_df['timestamp'].min() + 0.5)) & (
                cut_df['timestamp'] <= (cut_df['timestamp'].max() - 0.5))]

        print(cut_df)
        return np.array_split(cut_df, 3)


    def calc_frequencies(self, dataframe):
        print(dataframe)

        SAMLPE_LENGTH = round(dataframe['timestamp'].max() - dataframe['timestamp'].min(), 4)

        filtered_df = dataframe.copy()
        filtered_df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']] = filtered_df[
            ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']].clip(lower=0.05)

        print(filtered_df)

        kernel = signal.gaussian(10, 3)
        kernel /= np.sum(kernel)

        gaussian_df = filtered_df.copy()
        gaussian_df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']] = gaussian_df[
            ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']].transform(
            lambda x: np.convolve(x, kernel, 'same'), raw=True)

        print(gaussian_df)

        row = {}
        sum_sensors = 0
        for sensor in config.SENSOR_NAMES:
            spectrum = np.abs(np.fft.fft(gaussian_df[sensor]))
            frequencies = np.fft.fftfreq(len(gaussian_df[sensor]), 1 / config.SAMPLING_RATE)
            mask = frequencies > 0
            frequency = np.argmax(spectrum[mask] * config.SAMPLING_RATE) / SAMLPE_LENGTH
            row[sensor] = frequency
            sum_sensors += frequency
        row['label'] = gaussian_df['label'].values[1]
        row['sum'] = sum_sensors

        self.train_data_list.append(row)

    def split_train_test(self):
        self.train_data, self.test_data = model_selection.train_test_split(self.train_data.copy())
        #self.test_data = self.train_data.copy()

    def fit_classifier(self):
        print("fit classifier on train data...")
        classifier_linear = svm.SVC(kernel='linear')
        classifier_poly = svm.SVC(kernel='poly')
        classifier_rbf = svm.SVC(kernel='rbf')
        # train all three classifiers
        #x_train = self.train_data[config.SENSOR_NAMES]
        x_train = self.train_data[['sum']]
        classifier_linear.fit(x_train, self.train_data['label'])
        classifier_poly.fit(x_train, self.train_data['label'])
        classifier_rbf.fit(x_train, self.train_data['label'])
        return classifier_linear, classifier_poly, classifier_rbf

    def evaluate(self, classifier_linear, classifier_poly, classifier_rbf):
        #x_test = self.test_data[config.SENSOR_NAMES]
        x_test = self.test_data[['sum']]
        # run prediction on test data
        predictions_linear = classifier_linear.predict(x_test)
        predictions_poly = classifier_poly.predict(x_test)
        predictions_rbf = classifier_rbf.predict(x_test)
        accuracy_linear = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_linear)
        accuracy_poly = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_poly)
        accuracy_rbf = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_rbf)
        max_accuracy = max([accuracy_linear, accuracy_poly, accuracy_rbf])
        if max_accuracy == accuracy_linear:
            print(f'LINEAR classifier has higher accuracy than poly({accuracy_poly}) or rbf({accuracy_rbf}): {accuracy_linear}')
            self.classifier = classifier_linear
        elif max_accuracy == accuracy_poly:
            print(f'POLY classifier has higher accuracy than linear({accuracy_linear}) or rbf({accuracy_rbf}): {accuracy_poly}')
            self.classifier = classifier_poly
        elif max_accuracy == accuracy_rbf:
            print(f'RBF classifier has higher accuracy than poly({accuracy_poly}) or linear({accuracy_linear}): {accuracy_rbf}')
            self.classifier = classifier_rbf

    def train(self):
        self.read_data()
        self.split_train_test()
        classifier_linear, classifier_poly, classifier_rbf = self.fit_classifier()
        self.evaluate(classifier_linear, classifier_poly, classifier_rbf)
        print("training done")
