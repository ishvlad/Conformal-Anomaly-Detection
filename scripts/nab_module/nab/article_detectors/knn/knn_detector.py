from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 1
        self.dim = 1

        # Algorithm attributes
        self.buf = []
        self.training = []
        self.record_count = 0

        # Mahalanobis attributes
        self.sigma = np.diag(np.ones(self.dim))
        self.sigma_inv = np.diag(np.ones(self.dim))
        self.mean = -1

        self.rang = 0
        self.min_value = np.inf
        self.max_value = -np.inf

    def handleRecord(self, inputData):
        """
        inputRow = [inputData["timestamp"], inputData["value"]]
        """
        value = inputData["value"]
        self.buf.append(value)

        # TODO: consider quantiles instead of strict borders

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
            self.training.append(new_item)
            if self.record_count < self.probationaryPeriod - self.dim:
                return [0.0]
            else:
                self.update_sigma()

                return [self.get_NN_dist(np.array(new_item), np.array(self.training))]
