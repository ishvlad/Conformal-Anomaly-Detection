from nab.detectors.base import AnomalyDetector

import numpy as np
import math
import heapq

class KnnDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(KnnDetector, self).__init__(*args, **kwargs)

        self.buf = []
        self.training = []
        self.pred = 0
        self.max_dist = 0
        self.record_count = 0
        self.k = 43
        self.dim = 4
        self.sigma = np.diag(np.ones(self.dim))
        self.rang = self.inputMax - self.inputMin
            
    def metric(self,a,b):
        diff = a-np.array(b)
        return np.dot(np.dot(diff,self.sigma),diff.T)
    
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
        return sum(dists)/self.rang

    def sigmoid(self, x):
        return np.nan_to_num(x/(1+x))

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
                ost = self.record_count % self.probationaryPeriod
                if ost == 0 or ost == int(self.probationaryPeriod/2):
                    try:
                        self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
                    except np.linalg.linalg.LinAlgError:
                        print('Singular Matrix at record', self.record_count)
                
                if self.pred > 0:
                    self.pred -= 1
                    return [0.5]
                
                res = self.sigmoid(self.get_NN_dist(new_item))
                
                
                if res > 0.995:
                    self.pred = int(self.probationaryPeriod/5)
            
                return [res]
