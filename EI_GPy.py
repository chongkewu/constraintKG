# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:16:34 2018

@author: 44266
"""
from pyDOE import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scop
from mpl_toolkits.mplot3d import Axes3D
import GPy
from scipy.stats import norm
import logging
import datetime
def main():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='EI_GPy.log',\
                        format='%(message)s',\
                        filemode='w', level=logging.DEBUG)
    
    # Initialize
    num = 5
    re = lhs(2,samples = num)*4-2
    z = np.zeros(shape=num)
    for i in range(num):
        z[i] =  scop.rosen(re[i,:])
    z = z.reshape(z.size,1)
    A_set = get_A_set(num=50)
    
# =============================================================================
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x,y,z)
#     return
# =============================================================================
    # EI sampling
    num_sample = 200
    for i in range(num_sample):
        # update model
        ker = GPy.kern.Matern52(2,ARD=True)
        m = GPy.models.GPRegression(re,z,ker,noise_var= 1e-6)
        m.Gaussian_noise.fix()
        m.optimize(messages=False,optimizer = 'scg')
        # prediction
        mu_set,sig_set = m.predict(A_set)
        f_star = np.amin(z)
        ind_star = np.argmin(z)
        # recommendation
        EI_set = -(mu_set-f_star)*norm.cdf(-(mu_set-f_star)/sig_set)+sig_set*\
        norm.pdf(-(mu_set-f_star)/sig_set)
        ind = np.argmax(EI_set)
        sample_next = A_set[ind]
        # sampling
        f_next = scop.rosen(sample_next)
        f_next = np.array([[f_next]])
        z = np.concatenate((z,f_next),axis=0)
        re = np.concatenate((re,np.array([sample_next])),axis=0)
        currentDT = datetime.datetime.now()
        
        logging.info('Current minimnum is %f in location',f_star)
        logging.info(re[ind_star])
        logging.info('the %d th sample point is:',(i+1))
        logging.info(sample_next)
        logging.info('EI value is: %d',EI_set[ind])
        logging.info('sample value is %f',f_next)
        logging.info(currentDT.strftime("%Y-%m-%d %H:%M:%S")+'\n')
        
        print("Current minimnum is %f in location"%f_star)
        print(re[ind_star])
        print("the %d th sample point is:"%(i+1))
        print(sample_next)
        print("EI value is: %d"%EI_set[ind])
        print("sample value is %f"%f_next)
        print (currentDT.strftime("%Y-%m-%d %H:%M:%S")+'\n')
        
def get_A_set(num=50,min=-2,max=2):
    # get the grid coordinates
    nx, ny = (num, num)
    x = np.linspace(min, max, nx)
    y = np.linspace(min, max, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(xv.size,1)
    yv = yv.reshape(yv.size,1)
    A_set = np.concatenate((xv,yv),axis=1)
    return A_set
    
if __name__ == "__main__":
    main()