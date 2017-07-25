# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:59:24 2017

@author: Damon Lee
"""

from datetime import datetime
import numpy as np
import sys
from Layer import RNNLayer
from operations import softmax


class Model:
    
    #word_dim= word vector 的長度 , hidden_dim=S 的長度(可自訂)
    def __init__(self,word_dim,hidden_dim,bptt_truncate=4):
        self.word_dim=word_dim
        self.hidden_dim=hidden_dim
        self.bptt_truncate=bptt_truncate
        self.U=np.random.uniform(-np.sqrt(1.0/word_dim),np.sqrt(1.0/word_dim),(hidden_dim,word_dim))
        self.W=np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim))
        self.V=np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(word_dim,hidden_dim))
    
        
    # example: x=[0,179,243,17] y=[179,243,17,1]    
    def forward_propagation(self,x):
        Steps=len(x)
        layers=[]
        prev_s=np.zeros(self.hidden_dim)
        for step in range(Steps):
            layer=RNNLayer()
            input_vec=np.zeros(self.word_dim)
            input_vec[x[step]]=1
            layer.forward(input_vec,prev_s,self.U,self.W,self.V)
            prev_s=layer.s
            layers.append(layer)
        return layers
        
        
    def predict(self,x):
        output=softmax()
        layers=self.forward_propagation(x)
        return [np.argmax(output.predict(layer.mulv)) for layer in layers]


    #mulv= softmax之前的 y^ 
    def calculate_loss(self, x, y):
        assert len(x)==len(y)
        output=softmax()
        layers=self.forward_propagation(x)
        loss=0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulv,y[i])
        
        return loss/float(len(y))        
        
    def calculate_total_loss(self,X,Y):
        loss=0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i],Y[i])
        return loss/float(len(Y))
        
    def bptt(self,x,y):
        assert len(x)==len(y)
        output=softmax()
        layers=self.forward_propagation(x)
        dU=np.zeros(self.U.shape)
        dW=np.zeros(self.W.shape)
        dV=np.zeros(self.V.shape)
        
        Steps=len(layers)
        prev_s_t=np.zeros(self.hidden_dim)
        diff_s=np.zeros(self.hidden_dim)
        for step in range(Steps):
            dmulv=output.diff(layers[step].mulv,y[step])
            input_vec=np.zeros(self.word_dim)
            input_vec[x[step]]=1
            dprev_s, dU_t, dW_t, dV_t = layers[step].backward(input_vec, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)
            prev_s_t=layers[step].s

            dmulv=np.zeros(self.word_dim)
            for i in range(step-1, max(-1, step-self.bptt_truncate-1), -1):
                input_vec = np.zeros(self.word_dim)
                input_vec[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(input_vec, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
                dU_t += dU_i
                dW_t += dW_i
            dU += dU_t
            dV += dV_t
            dW += dW_t
                
        return (dU,dW,dV)


    def sgd_step(self,x,y,learning_rate):
        dU,dW,dV=self.bptt(x,y)
        self.U -= learning_rate*dU
        self.W -= learning_rate*dW
        self.V -= learning_rate*dV
        
    def train(self, X,Y,learning_rate=0.005,nepoch=100,evaluate_loss_after=5):
        num_example_seen=0
        losses=[]
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after==0):
                loss=self.calculate_total_loss(X,Y)
                losses.append((num_example_seen,loss))
                time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_example_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses)>1  and losses[-1][1] >losses[-2][1]:
                    learning_rate=learning_rate*0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
                
            # For each training example...     
            for i in range(len(Y)):
                self.sgd_step(X[i],Y[i],learning_rate)
                num_example_seen += 1 
        return losses


        
        
        