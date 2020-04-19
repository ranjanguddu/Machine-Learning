#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:55:13 2020

@author: vikasran
"""
import numpy as np
def unnormalize(mu, std, img):
    
  #print("unnormalized function get called. \n Shape of the Image is:{} \n Dimension of the Image is:{}".format(img.shape, img.ndimension()))
  img = img.numpy().astype(dtype=np.float32)
  
  for i in range(img.shape[0]):
    img[i] = (img[i]*std[i])+mu[i]
  #print("image normalized with mu ={} and std is:{}".format(mu,std))
  return np.transpose(img, (1,2,0))

