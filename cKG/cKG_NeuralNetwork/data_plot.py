# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:33:12 2019
This script plots the benchmark of Neural network. It require the
program save the recommend value since the evaluation of objective 
is time consuming. For the purpose of report, we will do more evaluation
at the process of Bayesian Optimization. It will increase the time
consuming, but it is worthwhile since we will use parallel evaluation 
during Bayesian Optimization.

@author: Chongke Wu
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append("..")
from cKG import obj_func

def main(repeat=14, num_p=80, func="rosen", tau=3000):
    util_ls = []
    for i in range(3,repeat):
        fname = "02142019/"  + func + "/exp_test" + "/output" + str(i)
        with open(fname + '/data.pkl', 'rb') as f:
            myPara, fD = pickle.load(f)        
        para = myPara.X_prd[myPara.rec_ind, :]
        c_val = obj_func(para, func)['c1']
        util = obj_func(para, func)['f'] * (c_val >= 0) + tau * (c_val < 0)
        util_ls.append(util[0:num_p])
    util_arr = np.array(util_ls).T
    print(util_arr.shape)
    plt.figure()    
    x_mean = 2*myPara.num_train + np.arange(num_p)
    y_mean = np.mean(util_arr, axis=1)
    print(y_mean[0])
    y_err = stats.sem(util_arr, axis=1)
    with open(func+'_cKG.pkl', 'wb') as f:
        pickle.dump([x_mean, y_mean, y_err], f)
    plt.errorbar(x_mean, np.mean(util_arr, axis=1), stats.sem(util_arr, axis=1))
    plt.ylim(-1, 1000)
    plt.xlim(0, num_p)
def test():
    fname = 'debug_0/data.pkl'
    with open(fname , 'rb') as f:
        myPara, fD = pickle.load(f) 
    print(myPara.rec_util)
    print("objective rec value {}".format(myPara.obj_rec))
    print("constraint rec value {}".format(myPara.cons_rec))
    print("objective value {}".format(fD['f'].obs_val))
    print("constraint value {}".format(fD['c1'].obs_val))
if __name__ == "__main__":
    #main()
    test()