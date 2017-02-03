from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnlcdDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnlcdDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 1
        self.dim = 1

        # Algorithm attributes
        self.buf = []
        self.training = []
        self.record_count = 0
        self.rang = self.inputMax - self.inputMin

        # Mahalanobis attributes
        self.sigma = np.diag(np.ones(self.dim))
        self.sigma_inv = np.diag(np.ones(self.dim))
        self.mean = -1

        # Inductive attributes
        self.calibration = []
        self.calibration_scores = []

    def metric(self, a, b):
        diff = a - np.array(b)
        return np.dot(np.dot(diff, self.sigma_inv), diff.T) ** 0.5

    def get_NN_dist(self, item, array=None):
        if array is None:
            array = self.training
        if type(array) is not np.ndarray:
            array = np.array(array)

        if self.k == 1 and self.dim == 1:
            dists = np.abs(array - item) * np.sqrt(self.sigma_inv)
            dists = np.min(dists)
        else:
            dists = []
            for x in array:
                dist = self.metric(x, item)
                if len(dists) < self.k:
                    dists.append(dist)
                else:
                    i = np.argmax(dists)
                    if dists[i] > dist:
                        dists[i] = dist

        return np.sum(dists) / (self.rang * self.k * self.dim ** 0.5)

    def update_sigma(self):
        try:
            X = self.training - np.mean(self.training, axis=0)
            self.sigma_inv = np.linalg.inv(np.dot(X.T, X))
            self.sigma_inv /= np.linalg.norm(self.sigma_inv, axis=0)

        except np.linalg.linalg.LinAlgError:
            print('Singular Matrix at record', self.record_count)

    def ncm(self, item, array=None):
        return self.get_NN_dist(item, array)

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
                # fill training set
                self.training.append(new_item)
                return [0.0]
            else:
                if self.record_count == self.probationaryPeriod - self.dim:
                    self.update_sigma()
                    # Leave One Out
                    loo_ncm = lambda x: self.ncm(x[1], np.delete(self.training, x[0]))
                    self.calibration_scores = list(map(loo_ncm, enumerate(self.training)))

                new_ncm = self.ncm(new_item)
                anomaly_score = 1. * np.sum(np.array(self.calibration_scores) < new_ncm) / len(self.calibration_scores)

                if self.record_count >= 2 * (self.probationaryPeriod - self.dim):
                    self.training.pop(0)
                    self.training.append(self.calibration.pop(0))
                    self.update_sigma()

            self.calibration.append(new_item)
            self.calibration_scores.pop(0)
            self.calibration_scores.append(new_ncm)

            return [anomaly_score]