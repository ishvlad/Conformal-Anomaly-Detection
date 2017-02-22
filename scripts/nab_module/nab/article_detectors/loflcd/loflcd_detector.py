from nab.detectors.base import AnomalyDetector

import numpy as np
import math

class LoflcdDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(LoflcdDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 3
        self.dim = 1

        # Algorithm attributes
        self.buf = []
        self.training = []
        self.record_count = 0

        # Mahalanobis attributes
        self.sigma = np.diag(np.ones(self.dim))
        self.sigma_inv = np.diag(np.ones(self.dim))
        self.mean = -1

        # Inductive attributes
        self.calibration_ncms = []
        self.calibration_scores = []

        self.rang = 0
        self.min_value = np.inf
        self.max_value = -np.inf

    def lof(self, item, array):
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        mean_dist, neighbours = self.get_NN_dist(np.array(item), array, return_nn=True)

        lrd = min(1. / mean_dist, self.rang * 1e+5)
        lrds = [min(1. / self.get_NN_dist(np.array(nn), array), self.rang * 1e+5) for nn in array[neighbours]]
        return np.sum(lrds) / self.k / lrd

    def ncm(self, item):
        return self.lof(item, np.array(self.training))

    def handleRecord(self, inputData):
        """
        inputRow = [inputData["timestamp"], inputData["value"]]
        """
        value = inputData["value"]
        self.buf.append(value)

        if value < self.min_value:
            self.min_value = value
            self.rang = self.max_value - self.min_value
        elif value > self.max_value:
            self.max_value = value
            self.rang = self.max_value - self.min_value

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
                training_ = np.array(self.training)
                if self.record_count == self.probationaryPeriod - self.dim:
                    self.update_sigma()
                    # Leave One Out
                    loo_ncm = lambda x: self.ncm(x)
                    self.calibration_scores = list(map(loo_ncm, training_))
                new_item = np.array(new_item)
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
