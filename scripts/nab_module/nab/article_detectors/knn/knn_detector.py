from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 5
        self.dim = 13

        # Attributes
        self.buf = []
        self.training = []
        self.record_count = 0
        self.sigma = np.diag(np.ones(self.dim))
        self.rang = self.inputMax - self.inputMin


    def metric(self, a, b):
        diff = a - np.array(b)
        return np.dot(np.dot(diff, self.sigma), diff.T) ** 0.5

    def get_NN_dist(self, item):
        dists = []
        for x in self.training[:-1]:
            dist = self.metric(x, item)
            if len(dists) < self.k:
                dists.append(dist)
            else:
                i = np.argmax(dists)
                if dists[i] > dist:
                    dists[i] = dist
        return sum(dists) / (self.rang * self.k * self.dim ** 0.5)

    def update_sigma(self):
        try:
            X = self.training - np.mean(self.training, axis=0)
            self.sigma = np.linalg.inv(np.dot(X.T, X))
            self.sigma /= np.linalg.norm(self.sigma, axis=0)
        except np.linalg.linalg.LinAlgError:
            print('Singular Matrix at record', self.record_count)

    def handleRecord(self, inputData):
        """
        inputRow = [inputData["timestamp"], inputData["value"]]
        """
        self.buf.append(inputData["value"])

        if len(self.buf) < self.dim:
            return [0.0]
        else:
            new_item = self.buf[-self.dim:]
            self.record_count += 1
            if self.record_count < self.probationaryPeriod - self.dim:
                self.training.append(new_item)
                return [0.0]
            else:
                self.training.append(new_item)

                self.update_sigma()

                return [self.get_NN_dist(new_item)]
