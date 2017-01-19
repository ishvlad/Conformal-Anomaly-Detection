from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnicadDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnicadDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 1
        self.dim = 6

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
        self.from_train_to_cal = 0.8 * (self.probationaryPeriod - self.dim)
        self.calibration_ncms = []

    def metric(self, a, b):
        diff = a - np.array(b)
        return np.dot(np.dot(diff, self.sigma_inv), diff.T) ** 0.5

    def get_NN_dist(self, item):
        if self.k == 1 and self.dim == 1:
            dists = np.abs(self.training - item) * np.sqrt(self.sigma_inv)
            dists = np.min(dists)
        else:
            dists = []
            for x in self.training:
                dist = self.metric(x, item)
                if len(dists) < self.k:
                    dists.append(dist)
                else:
                    i = np.argmax(dists)
                    if dists[i] > dist:
                        dists[i] = dist

        return np.sum(dists) / (self.rang * self.k * self.dim ** 0.5)

    def update_sigma(self, new_item, inverse=False):
        try:
            if inverse:
                self.mean = np.mean(self.training, axis=0).reshape(-1, 1)
                X = self.training - self.mean.T
                self.sigma = np.dot(X.T, X)
            else:
                delta_ = np.array([new_item]) - self.mean.T
                self.mean += delta_.T / self.record_count

                U = np.dot(delta_.T, delta_)
                U -= U / self.record_count
                self.sigma += U

            self.sigma_inv = np.linalg.inv(self.sigma)
            self.sigma_inv /= np.linalg.norm(self.sigma_inv, axis=0)
        except np.linalg.linalg.LinAlgError:
            print('Singular Matrix at record', self.record_count)

    def ncm(self, item):
        return self.get_NN_dist(item)

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

            if self.record_count < self.from_train_to_cal:
                # fill training set
                self.training.append(new_item)
                return [0.0]

            elif self.record_count < self.probationaryPeriod - self.dim:
                if type(self.training) is not np.array:
                    self.training = np.array(self.training)

                # fill calibration set
                self.update_sigma(new_item, inverse=(type(self.mean) is int and self.mean == -1))
                self.calibration_ncms.append(self.ncm(new_item))
                return [0.0]
            else:
                self.update_sigma(new_item)
                new_ncm = self.ncm(new_item)

                result = 1. * np.sum(np.array(self.calibration_ncms) < new_ncm) / len(self.calibration_ncms)
                self.calibration_ncms.append(new_ncm)

                return [result]