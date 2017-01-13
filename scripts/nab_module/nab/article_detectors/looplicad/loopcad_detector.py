from nab.detectors.base import AnomalyDetector

import numpy as np
import math

class LoopcadDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(LoopcadDetector, self).__init__(*args, **kwargs)
        
        self.buf = []
        self.training = []
        self.max_dist = 0
        self.record_count = 0
        self.lamb = 3
        self.k = 30
        self.dim = 20
        self.sigma = np.diag(np.ones(self.dim))
        self.rang = self.inputMax - self.inputMin
        self.probationaryPeriod = min(400, self.probationaryPeriod)
        self.ncms = []
        self.calibration = []

    def metric(self,a,b):
        diff = a-np.array(b)
        return np.dot(np.dot(diff,self.sigma),diff.T)
    
    def get_NN_dist(self, item, item_in_array=False, index=None):
        if index is not None:
            res = 0.
            for i in index:
                res += self.metric(item, self.training[i])
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
       
    def get_pdist(self, x, index=None):
        return self.lamb*(self.get_NN_dist(x, index=index)/self.k)**0.5
    
    
    def ncm(self,item, item_in_array=False):
        nneight_index = list(map(lambda x: self.training.index(x), self.get_NN(item)))
        self.Pdist = np.append(self.Pdist, self.get_pdist(item, index=nneight_index))[1:]
        new_Plof = self.k*self.Pdist[-1]/np.sum(self.Pdist[nneight_index])-1
        self.Plof=np.append(self.Plof,new_Plof)[1:]
        nPlof = self.lamb*((np.sum(self.Plof**2)/len(self.training))**0.5)
        
        return max(0, math.erf(self.Plof[-1]/nPlof/(2**0.5)))
        

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
                
                if(len(self.ncms) == 0):
                    self.Pdist = np.array(list(map(lambda x: self.get_pdist(x), self.training)))
                    self.Plof = np.empty(len(self.training))
                    self.ncms = np.empty(len(self.training))
                    for i,v in enumerate(self.training):
                        nneight_index = list(map(lambda x: self.training.index(x), self.get_NN(v)))
                        self.Plof[i] = self.k*self.Pdist[i]/np.sum(self.Pdist[nneight_index])-1
                    nPlof = self.lamb*((np.sum(self.Plof**2)/len(self.training))**0.5)
                    self.ncms = [max(0, math.erf(self.Plof[i]/nPlof/(2**0.5))) for i in range(len(self.training))]
                    
                new_ncm = self.ncm(new_item)
                result = 1.*len(np.where(np.array(self.ncms) < new_ncm)[0])/len(self.ncms)

                if self.record_count >= 2*self.probationaryPeriod:
                    self.training.pop(0)
                    self.training.append(self.calibration.pop(0))

                self.ncms.pop(0) 
                self.calibration.append(new_item)
                self.ncms.append(new_ncm)
            
                return [result]