from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnsliceDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnsliceDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 1
        self.dim = 1

        # Algorithm attributes
        self.buf = []
        self.training = []
        self.record_count = 0
        self.rang = self.inputMax - self.inputMin

        # Mahalanobis attributes
        self.sigma_inv = np.diag(np.ones(self.dim))

    def metric(self, a, b):
        diff = a - np.array(b)
        return np.sqrt(np.dot(np.dot(diff, self.sigma_inv), diff.T))

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
            self.sigma_inv = np.linalg.inv(np.dot(X.T, X))
            self.sigma_inv /= np.linalg.norm(self.sigma_inv, axis=0)

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
                self.training.pop(0)
                self.update_sigma()

                return [self.get_NN_dist(new_item)]