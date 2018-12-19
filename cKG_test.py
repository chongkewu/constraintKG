# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:40:45 2018
This script is unittest for cKG.py
We compare the predict method with the GPy GPCoregionalizedRegression method. 
The outpout mean and variance from both method should be equal. 
@author: 44266
"""
import unittest
import GPy
import numpy as np
from pprint import pprint
from cKG import predict_raw,init_x,rosen_constraint
import numpy.testing as npt
from paramz import ObsAr
from numpy.linalg import inv
from GPy.util.linalg import dtrtrs
class Test_predict(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        X_prd = np.array([[1.5,-1],[-0.5,0.2],[0,0.4],[1,1.5]])
        X1,self.X_prd_f,self.noise_dict_f = init_x(X_prd,2,h=0)    
        X2,self.X_prd_c,self.noise_dict_c = init_x(X_prd,2,h=1)
        self.pred_var = ObsAr(np.vstack([self.X_prd_f[0:2][:],self.X_prd_c[0:2][:]]))
        self.Xnew = self.X_prd_f
        obj = (rosen_constraint(X1)['f'][:,None])
        cons = (rosen_constraint(X2)['c1'][:,None])
        self.Y = np.vstack([obj,cons])
        K = GPy.kern.Matern52(input_dim=2, ARD = True)
        icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
        self.m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)        
        self.m['.*Mat52.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.        
        self.m.optimize()        
        self.m.mixed_noise.Gaussian_noise_0.variance=0
        self.m.mixed_noise.Gaussian_noise_1.variance=0        
        
        
    def test_predict_raw(self):
        mean_prd_f,var_prd_f = self.m.predict(self.X_prd_f,Y_metadata=self.noise_dict_f)
        mean_prd_c,var_prd_c = self.m.predict(self.X_prd_c,Y_metadata=self.noise_dict_c)         
        
        print(mean_prd_f,var_prd_f)
        K = self.m.posterior._K
        K_inv = inv(K)
        M1 = self.m.posterior.woodbury_vector
        M2 = np.matmul(K_inv,self.Y)
        npt.assert_array_almost_equal(M1,M2,decimal = 1)
        Kx = self.m.kern.K(self.pred_var,self.Xnew)
        mu1 = np.dot(Kx.T, M1)
        mu2 = np.matmul(Kx.T,M2)
        npt.assert_array_almost_equal(mu1,mu2,decimal = 4)
        tmp = dtrtrs(self.m.posterior._woodbury_chol, Kx)[0]
        tmp1 = np.square(tmp).sum(0)
        temp1 = np.matmul(K_inv,Kx)
        temp2 = np.sum(Kx.T*temp1.T,axis = 1)
        Kxx = self.m.kern.Kdiag(self.Xnew)
        npt.assert_array_almost_equal(tmp1,temp2,decimal = 4)
        
        mean,var = predict_raw(self.m,self.pred_var,self.Xnew,self.Y,K_inv,fullcov=False)
        print(mean,var)
        npt.assert_array_almost_equal(var_prd_f,var,decimal = 4)
    def tearDown(self):
        print('tearDown...')
        
        
if __name__ == '__main__':
    unittest.main()