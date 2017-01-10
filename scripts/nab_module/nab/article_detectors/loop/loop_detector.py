from nab.detectors.base import AnomalyDetector
import numpy as np
import math


class LoopDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        # Initialize the parent
        super(LoopDetector, self).__init__(*args, **kwargs)

        self.buf = []
        self.training = []
        self.pred = 0
        self.max_dist = 0
        self.record_count = 0
        self.lamb = 3
        self.k = 10
        self.dim = 3
        self.Pdist = []
        self.Plof = []
        self.sigma = np.diag(np.ones(self.dim))
        self.rang = self.inputMax - self.inputMin
        self.probationaryPeriod = min(400, self.probationaryPeriod)
            
    def metric(self,a,b):
        diff = a-np.array(b)
        return np.dot(np.dot(diff,self.sigma),diff.T)
    
    def get_pdist(self, x, index=None):
        return self.lamb*(self.get_NN_dist(x, index=None)/self.k)**0.5
    
    def get_NN_dist(self, item, item_in_array=True, index=None):
        if index is not None:
            res = 0.
            for i in index:
                res += self.training[i]
            return res/self.rang
        
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
    
    def get_NN(self, item, item_in_array=True):
        return sorted(self.training, key=lambda x:self.metric(x,item))[1 if item_in_array else 0:self.k+1]
    
    def dense(self, item, item_in_array=True):
        return self.k/max(0.001,self.get_NN_dist(item, item_in_array))

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
                
                if (len(self.Pdist) == 0):
                    self.Pdist = np.array(list(map(lambda x: self.get_pdist(x), self.training)))
                    self.Plof = np.empty(len(self.training))
                    for i,v in enumerate(self.training):
                        nneight_index = list(map(lambda x: self.training.index(x), self.get_NN(v)))
                        self.Plof[i] = self.k*self.Pdist[i]/np.sum(self.Pdist[nneight_index])-1
                else:
                    nneight_index = list(map(lambda x: self.training.index(x), self.get_NN(new_item)))
                    self.Pdist = np.append(self.Pdist, self.get_pdist(new_item, index=nneight_index))[1:]
                    new_Plof = self.k*self.Pdist[-1]/np.sum(self.Pdist[nneight_index])-1
                    self.Plof=np.append(self.Plof,new_Plof)[1:]
                
                nPlof = self.lamb*((np.sum(self.Plof**2)/len(self.training))**0.5)
                res = max(0, math.erf(self.Plof[-1]/nPlof/(2**0.5)))
                if res > 0.995:
                    self.pred = int(self.probationaryPeriod/5)
                
                return [res]
