# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:41:25 2018
This file is learn to use GPy package
@author: 44266
"""
import GPy
import numpy as np
from pyDOE import*
from pprint import pprint

def learn_model_basic():
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    x=np.random.uniform(-10,10,(15,1))
    y=2*np.sin(x)+.2*np.cos(4*x)
    # default noise_var is 1, which means the sample is not accurate.
    m = GPy.models.GPRegression(x,y,kernel,noise_var = 1)
    # the GPRegression parent is core.gp, many methods are used
    # optimize will change the kernel variance, lengthsacle and noise variance
    # we can fix some parameter: m.Gaussian_noise.fix()
    m.optimize(messages=True)
    # restarts will find a more accurate parameter
    # m.optimize_restarts(num_restarts = 10)
    # get the sample data
    # input should be like this:
# =============================================================================
#      X_S = np.arange(0,5,0.3)
#      X_S = X_S.reshape(X_S.size,1)
#      print(m.posterior_samples(X_S))
# =============================================================================
    # get the predict data
    print(m)
    #m.plot()
    X_S = np.arange(0,5,0.3)
    X_S = X_S.reshape(X_S.size,1)
    #print(m.predict(X_S))

if __name__ == "__main__":
    learn_model_basic()