import config


class Trainer:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self):
        # TODO
        print("reading train data...")
        data = []  # this is eig 1 dataframe, overall dataframe
        print("fit classifier on train data...")
        # classifier = svm.SVC(kernel='rbf') # non-linear classifier
        self.classifier.fit(data[config.SENSOR_NAMES], data['label'])
        print("done")
