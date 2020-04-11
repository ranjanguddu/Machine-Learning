#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:14:33 2020

@author: vikasran
"""

import matplotlib.pyplot as plt
import numpy as np
# if using a jupyter notebook
#%matplotlib inline 

def Cosine_Plot():
    plt.figure(figsize=(20,5))

    init = 0.5

    x = np.arange(0,10*np.pi,0.001)   # start,stop,step
    y = 1/10*(init + np.arccos(np.cos(x)))

    plt.plot(x,y)
    plt.show()
    

#Cosine_Plot()