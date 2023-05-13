"""
this module trains and evaluates classifiers on gathered data
"""
import os
import pandas as pd
import config
import numpy as np
from scipy import signal
from sklearn import model_selection, metrics, svm
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


class Trainer:
    def __init__(self):
        self.classifier = None
        self.train_data = None
        self.test_data = None
        self.train_data_list = []

    def read_data(self):
        """
        read datat from csv and store it in one dataframe for training
        """
        print("reading train data...")
        for filename in os.listdir(config.DATAPATH):
            file = os.path.join(config.DATAPATH, filename)
            if os.path.isfile(file):
                dataframe = pd.read_csv(file)
                split_data = self.split(dataframe)
                for s in split_data:
                    self.calc_frequencies(s)
        self.train_data = pd.DataFrame(self.train_data_list)

    def split(self, dataframe):
        """
        cut first and last 0.5 seconds of every data set, then split every data set into 3
        :param dataframe: gathered dataset for one activity (~10 seconds)
        :return: split dataframe
        """
        cut_df = dataframe.copy()
        cut_df = cut_df[(cut_df['timestamp'] >= (cut_df['timestamp'].min() + 0.5)) & (
                cut_df['timestamp'] <= (cut_df['timestamp'].max() - 0.5))]
        return np.array_split(cut_df, 3)

    def calc_frequencies(self, dataframe: pd.DataFrame):
        """
        calculate frequencies for each sensor data and store it in a central dataframe
        :param dataframe: gathered and split dataset for one activity (~3 seconds)
        :return:
        """
        # determine exact sample length
        sample_length = round(dataframe['timestamp'].max() - dataframe['timestamp'].min(), 4)

        # filter out very small values
        filtered_df = dataframe.copy()
        filtered_df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']] = filtered_df[
            ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']].clip(lower=config.THRESHOLD)

        # gaussian clean data
        kernel = signal.gaussian(10, 3)
        kernel /= np.sum(kernel)
        gaussian_df = filtered_df.copy()
        gaussian_df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']] = gaussian_df[
            ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'grav_x', 'grav_y', 'grav_z']].transform(
            lambda x: np.convolve(x, kernel, 'same'), raw=True)

        row = {}
        sum_sensors = 0
        #sum_sensor_frequencies = 0
        #sum_sensor_amplitudes = 0
        #sensor_amplitudes = []
        # calculate frequency for each sensor
        for sensor in config.SENSOR_NAMES:
            spectrum = np.abs(np.fft.fft(gaussian_df[sensor]))
            # calculate amplitude
            #amplitudes = (2 / (sample_length * config.SAMPLING_RATE)) * np.abs(spectrum)
            #amplitude = np.mean(amplitudes)
            #row[sensor+'_ampl'] = amplitude
            #sum_sensor_amplitudes += amplitude
            #sensor_amplitudes.append(amplitude)
            # calculate frequency
            frequencies = np.fft.fftfreq(len(gaussian_df[sensor]), 1 / config.SAMPLING_RATE)
            mask = frequencies > 0
            frequency = np.argmax(spectrum[mask] * config.SAMPLING_RATE) / sample_length
            row[sensor] = frequency
            #sum_sensor_frequencies += frequency
        row['label'] = gaussian_df['label'].values[1]
        row['sum'] = sum_sensors
        #row['sum_freq'] = sum_sensor_frequencies
        #row['sum_ampl'] = sum_sensor_amplitudes
        #mean_sensor_amplitudes = sum(sensor_amplitudes) / len(sensor_amplitudes)
        #row['sum_ampl'] = mean_sensor_amplitudes
        self.train_data_list.append(row)

    def split_train_test(self):
        """
        split training data into test and train to evaluate classifier
        """
        #todo check if gleichmäßiger split
        self.train_data, self.test_data = model_selection.train_test_split(self.train_data)
        #self.test_data = self.train_data.copy()

    def fit_classifier(self):
        """
        fit lienar, poly and rbf classifiers with frequency sums of training data
        :return: trained classifiers
        """
        print("fit classifier on train data...")
        classifier_linear = svm.SVC(kernel='linear')
        classifier_poly = svm.SVC(kernel='poly')
        classifier_rbf = svm.SVC(kernel='rbf')
        # train all three classifiers
        x_train = self.train_data[['sum']]
        #x_train = self.train_data[config.SENSOR_NAMES_W_AMPL].values
        #x_train = self.train_data[['sum_freq']].values
        #y_train = self.train_data[['label']].values
        #x_train = self.train_data[['sum_freq', 'sum_ampl']].values
        #for index, row in self.train_data.iterrows():
        #    print(row['label'], row['sum_freq'])
            #print(self.train_data['label'], x_train[index])
        #print(self.train_data['label'])
        #classifier_linear.fit(x_train, y_train)
        #classifier_poly.fit(x_train, y_train)
        #classifier_rbf.fit(x_train, y_train)
        classifier_linear.fit(x_train, self.train_data['label'])
        classifier_poly.fit(x_train, self.train_data['label'])
        classifier_rbf.fit(x_train, self.train_data['label'])
        return classifier_linear, classifier_poly, classifier_rbf

    def evaluate(self, classifier_linear, classifier_poly, classifier_rbf):
        """
        evaluate linear, poly and rbf classifiers
        :param classifier_linear: trained linear classifier
        :param classifier_poly: trained poly classifier
        :param classifier_rbf: trained rbf classifier
        """
        #x_test = self.test_data[config.SENSOR_NAMES_W_AMPL].values
        x_test = self.test_data[['sum']]
        #y_test = self.test_data[['label']].values
        #x_test = self.test_data[['sum_freq', 'sum_ampl']].values
        # run prediction on test data
        predictions_linear = classifier_linear.predict(x_test)
        predictions_poly = classifier_poly.predict(x_test)
        predictions_rbf = classifier_rbf.predict(x_test)
        # calculate accuracy for each of the three classifiers
        #accuracy_linear = metrics.accuracy_score(y_true=y_test, y_pred=predictions_linear)
        #accuracy_poly = metrics.accuracy_score(y_true=y_test, y_pred=predictions_poly)
        #accuracy_rbf = metrics.accuracy_score(y_true=y_test, y_pred=predictions_rbf)
        accuracy_linear = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_linear)
        accuracy_poly = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_poly)
        accuracy_rbf = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_rbf)
        # check which classifier has the highest accuracy, use this for predictions
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
        #for index, row in self.train_data.iterrows():
        #    print(row['label'], row['sum_freq'], row['sum_ampl'])
        #print(self.train_data)
        print("training done")
