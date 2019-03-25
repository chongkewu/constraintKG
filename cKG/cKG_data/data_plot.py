# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:33:12 2019

@author: 44266
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
if __name__ == "__main__":
    main()