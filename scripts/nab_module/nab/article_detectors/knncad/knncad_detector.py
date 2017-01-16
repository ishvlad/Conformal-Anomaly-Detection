from nab.detectors.base import AnomalyDetector

import numpy as np
import math
import heapq

class KnncadDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(KnncadDetector, self).__init__(*args, **kwargs)

        self.buf = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.results = []
        self.record_count = 0

        self.dim = 26
        self.k = 6

        self.sigma = np.diag(np.ones(self.dim))

            
    def metric(self,a,b):
        diff = a-np.array(b)
        return np.dot(np.dot(diff,self.sigma),diff.T)

    def ncm(self,item, item_in_array=False):
        arr = map(lambda x:self.metric(x,item), self.training)
        return np.sum(heapq.nsmallest(self.k+item_in_array,arr))

    def handleRecord(self, inputData):
        """
        inputRow = [inputData["timestamp"], inputData["value"]]
        """
        self.buf.append(inputData["value"])
        self.record_count += 1
        
        if len(self.buf) < self.dim:
            return [0.0]
        else:
            new_item = self.buf[-self.dim:]
            if self.record_count < self.probationaryPeriod:
                self.training.append(new_item)
                return [0.0]
            else:
                ost = self.record_count % self.probationaryPeriod
                if ost == 0 or ost == int(self.probationaryPeriod/2):
                    try:
                        self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
                    except np.linalg.linalg.LinAlgError:
                        print('Singular Matrix at record', self.record_count)
                if len(self.scores) == 0:
                    self.scores = list(map(lambda v: self.ncm(v, True), self.training))
                    self.results = list(map(lambda v: 1.*len(np.where(np.array(self.scores) < v)[0])/len(self.scores), self.scores))
                    
                new_score = self.ncm(new_item)
                result = 1.*len(np.where(np.array(self.scores) < new_score)[0])/len(self.scores)
                
                if self.record_count >= 2*self.probationaryPeriod:
                    if self.results[0] < self.append:
                        self.training.pop(0)
                        self.training.append(self.calibration.pop(0))
                    else:
                        self.calibration.pop(0)
                
                self.scores.pop(0)    
                self.results.pop(0)
                self.calibration.append(new_item)
                self.scores.append(new_score)
                self.results.append(result)

                return [result]
