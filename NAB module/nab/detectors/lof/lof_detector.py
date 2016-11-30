from nab.detectors.base import AnomalyDetector
import numpy as np


class LofDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        # Initialize the parent
        super(LofDetector, self).__init__(*args, **kwargs)

        self.buf = []
        self.training = []
        self.pred = 0
        self.max_dist = 0
        self.record_count = 0
        self.denses = []
        self.k = 33
        self.dim = 5
        self.sigma = np.diag(np.ones(self.dim))
        self.rang = self.inputMax - self.inputMin
        self.probationaryPeriod = min(400, self.probationaryPeriod)

    def metric(self,a,b):
        diff = a-np.array(b)
        return np.dot(np.dot(diff,self.sigma),diff.T)
    
    def get_NN_dist(self, item, item_in_array=False):
        dists = []
        for x in self.training:
            dist = self.metric(x, item)
            if len(dists) < self.k+item_in_array:
                dists.append(dist)
            else:
                i = np.argmax(dists)
                if dists[i] > dist:
                    dists[i] = dist
        return sum(dists)/self.rang
    
    def get_NN_dense(self, item, item_in_array=False):
        dists = []
        for i,x in enumerate(self.training):
            dist = self.metric(x, item)
            if len(dists) < self.k+item_in_array:
                dists.append([dist, self.denses[i]])
            else:
                idx = np.argmax(dists,axis=0)[0]
                if dists[idx][0] > dist:
                    dists[idx] = [dist, self.denses[i]]
        return np.sum(dists, axis=0)[1]
    
    def dense(self, item, item_in_array=False):
        return self.k/max(0.001,self.get_NN_dist(item, item_in_array))

    def handleRecord(self, inputData):
        """
        inputRow = [inputData["timestamp"], inputData["value"]]
        """
        self.buf.append(inputData['value'])
        
        if len(self.buf) < self.dim:
            return [0.0]
        else:
            new_item = self.buf[-self.dim:]
            self.record_count += 1
            if self.record_count < self.probationaryPeriod - self.dim:
                self.training.append(new_item)
                return [0.0]
            else:
                ost = self.record_count % self.probationaryPeriod
                if ost == 0 or ost == int(self.probationaryPeriod/2):
                    try:
                        self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
                    except np.linalg.linalg.LinAlgError:
                        print('Singular Matrix at record', self.record_count)
                
                if(len(self.denses) == 0):
                    self.denses = list(map(lambda x: self.dense(x, True), self.training))
                    
                new_dense = self.dense(new_item)
                lof = self.get_NN_dense(new_item)/new_dense/self.k
                res = 0 if lof < 1 else 1-1/lof
                
                self.training.append(new_item)
                self.denses.append(new_dense)
                self.training.pop(0)
                self.denses.pop(0)
                
                if self.pred > 0:
                    self.pred -= 1
                    return [0.5]
                
                if res > 0.995:
                    self.pred = int(self.probationaryPeriod/5)
            
                return [res]
