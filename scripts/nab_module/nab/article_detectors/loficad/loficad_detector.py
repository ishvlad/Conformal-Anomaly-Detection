from nab.detectors.base import AnomalyDetector

import numpy as np
import math

class LoficadDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(LoficadDetector, self).__init__(*args, **kwargs)
        # Hyperparams
        self.k = 1
        self.dim = 2

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
        self.from_train_to_cal = 0.7 * (self.probationaryPeriod - self.dim)
        self.calibration_ncms = []

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
                self.update_sigma()
                self.calibration_ncms.append(self.ncm(new_item))
                return [0.0]
            else:
                self.update_sigma()
                new_ncm = self.ncm(new_item)

                result = 1. * np.sum(np.array(self.calibration_ncms) < new_ncm) / len(self.calibration_ncms)
                self.calibration_ncms.append(new_ncm)

                return [result]
