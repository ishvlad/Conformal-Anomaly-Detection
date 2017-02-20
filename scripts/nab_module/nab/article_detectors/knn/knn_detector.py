from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 10
        self.dim = 10

        # Algorithm attributes
        self.buf = []
        self.training = []
        self.record_count = 0
        self.rang = self.inputMax - self.inputMin

        # Mahalanobis attributes
        self.sigma = np.diag(np.ones(self.dim))
        self.sigma_inv = np.diag(np.ones(self.dim))
        self.mean = -1

    def metric(self, a, b):
        diff = a - np.array(b)
        return np.sqrt(np.dot(np.dot(diff, self.sigma_inv), diff.T))

    def update_sigma(self, new_item, inverse=False):
        if self.record_count == self.probationaryPeriod - self.dim:
            inverse = True

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
                self.update_sigma(new_item=new_item)

                return [self.get_NN_dist(np.array(new_item), np.array(self.training))]
