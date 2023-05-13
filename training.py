"""
this module trains and evaluates classifiers on gathered data
"""
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
        sum_frequencies = 0
        amplitude_acc_y = gaussian_df['acc_y'].max() - gaussian_df['acc_y'].min()
        # calculate frequency for each sensor
        for sensor in config.SENSOR_NAMES:
            spectrum = np.abs(np.fft.fft(gaussian_df[sensor]))
            # calculate frequency
            frequencies = np.fft.fftfreq(len(gaussian_df[sensor]), 1 / config.SAMPLING_RATE)
            mask = frequencies > 0
            frequency = np.argmax(spectrum[mask] * config.SAMPLING_RATE) / sample_length
            row[sensor] = frequency
            sum_frequencies += frequency
        row['label'] = gaussian_df['label'].values[1]
        row['sum'] = sum_frequencies
        row['amplitude_acc_y'] = amplitude_acc_y
        self.train_data_list.append(row)

    def split_train_test(self):
        """
        split training data into test and train to evaluate classifier
        """
        self.train_data, self.test_data = model_selection.train_test_split(self.train_data, test_size=0.1, train_size=0.9)

    def fit_classifier(self):
        """
        fit linear and poly classifiers with frequency sums of training data
        :return: trained classifiers
        """
        print("fit classifier on train data...")
        classifier_linear = svm.SVC(kernel='linear')
        classifier_poly = svm.SVC(kernel='poly')
        # train linear and poly classifiers
        x_train = self.train_data[['sum', 'amplitude_acc_y']]
        classifier_linear.fit(x_train, self.train_data['label'])
        classifier_poly.fit(x_train, self.train_data['label'])
        return classifier_linear, classifier_poly

    def evaluate(self, classifier_linear, classifier_poly):
        """
        evaluate linear, poly and rbf classifiers
        :param classifier_linear: trained linear classifier
        :param classifier_poly: trained poly classifier
        """
        x_test = self.test_data[['sum', 'amplitude_acc_y']]
        # run prediction on test data
        predictions_linear = classifier_linear.predict(x_test)
        predictions_poly = classifier_poly.predict(x_test)
        # calculate accuracy for linear and poly classifiers
        accuracy_linear = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_linear)
        accuracy_poly = metrics.accuracy_score(y_true=self.test_data['label'], y_pred=predictions_poly)
        # check which classifier has the highest accuracy, use this for predictions
        max_accuracy = max([accuracy_linear, accuracy_poly])
        if max_accuracy == accuracy_linear:
            print(f'LINEAR classifier has higher accuracy than poly({accuracy_poly})): {accuracy_linear}')
            self.classifier = classifier_linear
        elif max_accuracy == accuracy_poly:
            print(f'POLY classifier has higher accuracy than linear({accuracy_linear})): {accuracy_poly}')
            self.classifier = classifier_poly

    def train(self):
        self.read_data()
        self.split_train_test()
        classifier_linear, classifier_poly = self.fit_classifier()
        self.evaluate(classifier_linear, classifier_poly)
        print("training done")
