# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:40:45 2018
This script is unittest for cKG.py
We compare the predict method with the GPy GPCoregionalizedRegression method. 
The outpout mean and variance from both method should be equal. 
@author: 44266
"""
import unittest
import pickle
import numpy as np
import numpy.testing as npt
import time
from scipy.stats import norm
from GPy.util.linalg import jitchol
from GPy.util import diag
from copy import copy
from cKG import predict_chol, predict_chol_mean, CRN_gen, jitchol_plus ,Aver_Un_star_samples

class Test_predict_cholesky(unittest.TestCase):
    def setUp(self):
        print("begin test...")
    def test_predict_chol(self):
        fname = 'debug_0'
        with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
            myPara, fD, m, j = pickle.load(f)
        npt.assert_array_equal(m.X, myPara.pred_var)
        npt.assert_array_equal(m.Y, myPara.Y)        
        mean, var = m.predict(fD['f'].X_prd, Y_metadata=fD['f'].noise_dict)
        mean_chol, var_chol = predict_chol(m, fD['f'].X_prd)        
        npt.assert_array_equal(mean_chol, mean)
        npt.assert_array_equal(var_chol, var)
    def test_predict_chol_mean(self):
        fname = 'debug_0'
        with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
            myPara, fD, m, j = pickle.load(f)
        mean, _= m.predict(fD['f'].X_prd, Y_metadata=fD['f'].noise_dict)
        Ky = m.posterior._K.copy()
        diag.add(Ky, 1e-6*np.ones(Ky.shape[0])+1e-8)
        Lw = jitchol(Ky)
        Kx = m.kern.K(m.X, fD['f'].X_prd)
        Y = m.Y
        mean_chol = predict_chol_mean(Kx, Lw, Y)
        npt.assert_array_equal(mean, mean_chol)
    def test_get_un_star(self):
        fname = 'debug_0'
        with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
            myPara, fD, m, ind_j = pickle.load(f)
        spl_set = CRN_gen(fD, 'c1', myPara.spl_num)
        Pr_feasible = -norm.cdf(0, loc=fD['c1'].mean_prd, scale=np.sqrt(fD['c1'].var_prd))+1        
        un_set = np.zeros((myPara.num, 1))  
        E_n_condi = np.zeros((myPara.num, 1))
        tic = time.time()
        for j in range(myPara.num):
            spl_x1 = np.array([fD['c1'].X_prd[j]])
            spl_x2 = np.array([fD['f'].X_prd[j]])        
            Lw = jitchol_plus(m, myPara.pred_var, m.posterior._K, spl_x1, fD['f'].noise)
            pred_var = np.vstack([myPara.pred_var, spl_x1])
            Kx_plus = m.kern.K(pred_var, spl_x2)
            un_set, E_n_condi = Aver_Un_star_samples(un_set, E_n_condi, j, myPara, spl_set, Lw, Kx_plus, Pr_feasible, fD)        
        un_star = np.min(un_set)        
        print(time.time()-tic)
        assert(un_star>-200)
        ind = np.unravel_index(np.argmin(un_set, axis=None),un_set.shape)[0]
        dict_noise = copy(fD['f'].noise_dict)
        dict_noise['output_index'] = fD['f'].noise_dict['output_index'][ind, :]
        mean, var = m.predict(fD['f'].X_prd[ind,:][np.newaxis], Y_metadata=dict_noise)
    def tearDown(self):
        print("Test completed.")


if __name__ == '__main__':
    unittest.main()