# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:53:42 2017

@author: Damon Lee
"""

import numpy as np
from rnn import Model
from preprocessing import getSentenceData

word_dim=8000
hidden_dim=100
X_train,Y_train=getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
rnn=Model(word_dim,hidden_dim)

loss=rnn.train(X_train[:100],Y_train[:100],learning_rate=0.005, nepoch=10, evaluate_loss_after=1)
