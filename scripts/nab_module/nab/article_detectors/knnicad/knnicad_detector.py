from nab.detectors.base import AnomalyDetector

import numpy as np


class KnnicadDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(KnnicadDetector, self).__init__(*args, **kwargs)
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

        # Inductive attributes
        self.from_train_to_cal = 0.8 * (self.probationaryPeriod - self.dim)
        self.calibration_ncms = []

        self.rang = 0
        self.min_value = np.inf
        self.max_value = -np.inf



    def ncm(self, item):
        return self.get_NN_dist(item)

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

            if self.record_count < self.from_train_to_cal:
                # fill training set
                self.training.append(new_item)
                return [0.0]

            elif self.record_count < self.probationaryPeriod - self.dim:
                if type(self.training) is not np.array:
                    self.training = np.array(self.training)

                # fill calibration set
                self.update_sigma()
                self.calibration_ncms.append(self.ncm(new_item))
                return [0.0]
            else:
                self.update_sigma()
                new_ncm = self.ncm(new_item)

                result = 1. * np.sum(np.array(self.calibration_ncms) < new_ncm) / len(self.calibration_ncms)
                self.calibration_ncms.append(new_ncm)

                return [result]