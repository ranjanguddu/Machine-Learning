#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:09 2020

@author: vikasran
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:09 2020

@author: vikasran
"""

#import numpy as np
from Denorm import unnormalize
import matplotlib.pyplot as plt


def Display_Sample_Image(mu, std, data_loader):
    
    
    
    # print few testing images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()



    figure = plt.figure(figsize=(20, 20))
    figure.subplots_adjust(hspace=1, wspace=0.5)
    for i in range(25):
        im = images[i]
        lab = labels[i]
        plt.subplot(5,5,i+1)
        plt.imshow(unnormalize(mu, std, im), aspect='auto')
        plt.title("Label: %s" % lab)

    plt.show()

    