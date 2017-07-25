# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:48:49 2017

@author: Damon Lee
"""

import numpy as np


#gate
class MultipleGate:
    def forward(self,W,x):
        return np.dot(W,x)
    
    def backward(self,W,x,dz):
        dW=np.asarray(np.dot(np.transpose(np.asmatrix(dz)),np.asmatrix(x)))
        dx=np.dot(np.transpose(W),dz)
        return dW,dx
    
class AddGate:
    def forward(self,x1,x2):
        return x1+x2
    
    def backward(self,x1,x2,dz):
        dx1=dz*np.ones_like(x1)
        dx2=dz*np.ones_like(x2)
        return dx1,dx2
        
#activation        
class sigmoid:
    def forward(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    def backward(self,x,top_diff):
        output=self.forward(x)
        return (1.0-output)*output*top_diff

class Tanh:
    def forward(self,x):
        return np.tanh(x)
    
    def backward(self,x,top_diff):
        output=self.forward(x)
        return (1.0-np.square(output))*top_diff

#output
class softmax:
    def predict(self,x):
        exp_scores=np.exp(x)
        return exp_scores/np.sum(exp_scores)
        
    def loss(self,x,y):
        probs=self.predict(x)
        return -np.log(probs[y])
        
    def diff(self,x,y):  #y^-y    
        probs=self.predict(x)
        probs[y] -=1.0
        return probs
        
        
        



