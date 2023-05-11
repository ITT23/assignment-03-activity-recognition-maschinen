import os
import pandas as pd
import config
from sklearn import model_selection, metrics, svm


class Trainer:
    def __init__(self):
        self.classifier = None
        self.train_data = None
        self.test_data = None

    def read_data(self):
        print("reading train data...")
        dataframes = []
        for filename in os.listdir(config.DATAPATH):
            file = os.path.join(config.DATAPATH, filename)
            if os.path.isfile(file):
                dataframes.append(pd.read_csv(file))
        self.train_data = pd.concat(dataframes)

    def split_train_test(self):
        self.train_data, self.test_data = model_selection.train_test_split(self.train_data.copy())

    def fit_classifier(self):
        print("fit classifier on train data...")
        classifier_linear = svm.SVC(kernel='linear')
        classifier_poly = svm.SVC(kernel='poly')
        classifier_rbf = svm.SVC(kernel='rbf')
        # train all three classifiers
        # todo: test if required for each row in dataframe
        x_train = [self.train_data[sensor] for sensor in config.SENSOR_NAMES]
        classifier_linear.fit(x_train, self.train_data['label'])
        classifier_poly.fit(x_train, self.train_data['label'])
        classifier_rbf.fit(x_train, self.train_data['label'])
        return classifier_linear, classifier_poly, classifier_rbf

    def evaluate(self, classifier_linear, classifier_poly, classifier_rbf):
        # todo: test if required for each row in dataframe
        x_test = [self.test_data[sensor] for sensor in config.SENSOR_NAMES]

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
