#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:09 2020

@author: vikasran
"""

import numpy as np
from Denorm import unnormalize
import matplotlib.pyplot as plt


def Display_Sample_Image(mu, std, train_loader):
    
    
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape) 

    num_classes = 10
# display 10 images from each category. 
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    r, c = 10, 11
    n = 9
    fig = plt.figure(figsize=(14,14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    for i in range(num_classes):
        idx = np.random.choice(np.where(labels[:]==i)[0], n)
        ax = plt.subplot(r, c, i*c+1)
        ax.text(-1.5, 0.5, class_names[i], fontsize=14)
        plt.axis('off')
        for j in range(1, n+1):
            plt.subplot(r, c, i*c+j+1)
            
            plt.imshow(unnormalize(mu, std,images[idx[j-1]]), interpolation='none')
            plt.axis('off')
    plt.show()