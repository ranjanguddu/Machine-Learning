#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:06:10 2020

@author: vikasran
"""

# calculation of mean and standard deviation
#from tqdm import tqdm_notebook
#from PIL import Image

import numpy as np

class Cal_Mean_STD():
    def cal_stats(train):
        
        n = 0
        s = np.zeros(3)
        sq = np.zeros(3)
        for data, l in train: # to show the prgress bar put tqdm_notebook(train) instead of (train)
            
            x = np.array(data)/255
            s += x.sum(axis=(0,1))
            sq += np.sum(np.square(x), axis=(0,1))
            n += x.shape[0]*x.shape[1]
            
        mu = s/n
        std = np.sqrt((sq/n - np.square(mu)))
        return mu, std
        
        
        
        




