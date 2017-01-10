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
        self.pred = -1
        self.success = 0
        
        self.append = 0.83
        self.dim = 26
        self.k = 6
        self.success_border = 5

        self.sigma = np.diag(np.ones(self.dim))

        # 0.84 19  5  4 | 0.9970, 41.27, 59.45, 53.40

        # appe|dm| k|sb
        # 0.76 26  6  5 | 0.9960, 31.87, 46.94, 41.96
        # 0.83 26  6  5 | 
        # 0.90 26  6  5 | 
        # 0.97 26  6  5 | 
        #
        #
        # 0.70  5 14  3 | 
        # 0.70 12 14  3 | 
        # 0.70 20 14  3 | 
        # 0.70 27 14  3 | 
        #
        #
        # 0.97 16  5  6 | 
        # 0.97 16 12  6 | 
        # 0.97 16 20  6 |
        # 0.97 16 27  6 |
        #
        #
        # 0.92 11 26  2 |
        # 0.92 11 26  4 |
        # 0.92 11 26  6 |
        # 0.92 11 26  8 |



            
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
                
                if self.pred > 0:
                    self.pred -= 1
                    return [0.5]
                elif result >= 0.9965:
                    self.success += 1
                    if self.success == self.success_border:
                        self.pred = int(self.probationaryPeriod/5)
                        self.success = 0
                        return [result]
                    else:
                        return [0.5]
                else:
                    self.success = 0
                return [min(0.9965, result)]
