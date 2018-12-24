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
from cKG import *
from pyDOE import *
import numpy.testing as npt
from paramz import ObsAr
from numpy.linalg import inv,eig
from GPy.util.linalg import dtrtrs
import time
from scipy.stats import norm
from copy import copy

class Test_get_un_star(unittest.TestCase):
    def setUp(self):
        pass
    def test_get_un_star(self):
        pass
    def tearDown(self):
        pass
    
class Test_predict(unittest.TestCase):
    def setUp(self):
        print('setUp Model...')
        X_prd = np.array([[1.5,-1],[-0.5,0.2],[0,0.4],[1,1.5]])
        X1,self.X_prd_f,self.noise_dict_f = init_x(X_prd,2,h='f')    
        X2,self.X_prd_c,self.noise_dict_c = init_x(X_prd,2,h='c1')
        self.pred_var = ObsAr(np.vstack([self.X_prd_f[0:2][:],self.X_prd_c[0:2][:]]))        
        obj = (rosen_constraint(X1)['f'][:,None])
        cons = (rosen_constraint(X2)['c1'][:,None])
        self.Y = np.vstack([obj,cons])
        K = GPy.kern.Matern52(input_dim=2, ARD = True)
        icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
        self.m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)        
        self.m['.*Mat52.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.                
        self.m['.*noise_0.var'].constrain_fixed(0.)    
        self.m['.*noise_1.var'].constrain_fixed(0.)    
        self.m.optimize()        
        #self.m.mixed_noise.Gaussian_noise_0.variance=0
        #self.m.mixed_noise.Gaussian_noise_1.variance=0        
        K = self.m.posterior._K
        self.K_inv = inv(K)   
    def test_predict_compare_GPy(self):
        self.Xnew = self.X_prd_f    
        M1 = self.m.posterior.woodbury_vector
        M2 = np.matmul(self.K_inv,self.Y)
        npt.assert_array_almost_equal(M1,M2,decimal = 1)
        
        Kx = self.m.kern.K(self.pred_var,self.Xnew)
        mu1 = np.dot(Kx.T, M1)
        mu2 = np.matmul(Kx.T,M2)
        npt.assert_array_almost_equal(mu1,mu2,decimal = 4)
        
        tmp = dtrtrs(self.m.posterior._woodbury_chol, Kx)[0]
        tmp1 = np.square(tmp).sum(0)
        temp1 = np.matmul(self.K_inv,Kx)
        temp2 = np.sum(Kx.T*temp1.T,axis = 1)
        npt.assert_array_almost_equal(tmp1,temp2,decimal = 4)    
    def test_predict_raw_obj(self):
        # test mean and variance of objective
        mean_prd_f,var_prd_f = self.m.predict(self.X_prd_f,Y_metadata=self.noise_dict_f)
        self.Xnew = self.X_prd_f        
        mean,var = predict_raw(self.m,self.pred_var,self.Xnew,self.Y,self.K_inv,fullcov=False)        
        npt.assert_array_almost_equal(var_prd_f,var,decimal = 4)
    def test_predict_raw_cons(self):        
        # test mean and variance of constraint
        self.Xnew = self.X_prd_c
        mean_prd_c,var_prd_c = self.m.predict(self.X_prd_c,Y_metadata=self.noise_dict_c)         
        mean_c,var_c = predict_raw(self.m,self.pred_var,self.Xnew,self.Y,self.K_inv,fullcov=False)        
        npt.assert_array_almost_equal(mean_prd_c,mean_c,decimal=4)
        npt.assert_array_almost_equal(var_prd_c,var_c,decimal=4)        
    def test_predict_raw_full_cov(self):    
        # test full posterior covariance matrix
        self.Xnew = self.X_prd_f
        _,var_full_f = self.m.predict(self.X_prd_f,Y_metadata=self.noise_dict_f,full_cov=True)
        _,var_full_f1 = predict_raw(self.m,self.pred_var,self.Xnew,self.Y,self.K_inv,fullcov=True)        
        npt.assert_array_almost_equal(var_full_f,var_full_f1,decimal=1)    
    def test_predict_mixed_noise(self):        
        temp = self.m.posterior._woodbury_chol
        self.m.mixed_noise.Gaussian_noise_0.variance=10
        self.m.mixed_noise.Gaussian_noise_1.variance=1         
        self.m.posterior._woodbury_chol = temp # it shouldn't change
        
        self.Xnew = self.X_prd_f 
        mean,var = predict_raw(self.m,self.pred_var,self.Xnew,self.Y,self.K_inv,fullcov=False)
        mean,var = predict_mixed_noise(self.m, mean, var, full_cov=False, Y_metadata=self.noise_dict_f)
        mean1,var1 = self.m.predict(self.Xnew,Y_metadata=self.noise_dict_f)
        npt.assert_array_almost_equal(var,var1,decimal = 1)                
    def tearDown(self):
        for attr in ('m','X_prd_f','K_inv','X_prd_c','X_prd_f','Xnew','Y',\
                     'noise_dict_c','noise_dict_f','pred_var'):
            self.__dict__.pop(attr,None)
        
        print('tearDown')
class Test_woodbury_inv(unittest.TestCase):
    def setUp(self):
        matrixSize = 500
        A = np.random.rand(matrixSize,matrixSize)
        B_plus = np.matmul(A,A.T)+0.001*np.eye(matrixSize)    
        B = B_plus[0:-1,0:-1]       
        C = np.array([B_plus[-1,0:-1]])
        D = np.array([[B_plus[-1,-1]]])        
        B_t = np.hstack(map(np.vstack,[[B,C],[C.T,D]]))
        npt.assert_array_equal(B_t,B_plus)
        B_inv = np.linalg.inv(B)
        npt.assert_array_almost_equal(np.matmul(B,B_inv),np.eye(matrixSize-1))
        self.matrixSize,self.B,self.B_plus,self.C,self.D,self.B_inv = matrixSize,B,B_plus,C,D,B_inv        
    def test_woodbury_inv_setup(self):
        self.assertTrue(np.all(np.linalg.eigvals(self.B_plus) > 0))
        self.assertTrue(np.all(np.linalg.eigvals(self.B) > 0))        
    def test_woodbury_inv(self):    
        B_plus_inv = woodbury_inv(self.B_inv,self.C.T,self.C,self.D)
        npt.assert_array_almost_equal(np.matmul(B_plus_inv,self.B_plus),np.eye(self.matrixSize))
    def test_woodbury_inv_speed(self):
        num = 3
        tic = time.time()           
        for i in range(num):
            B_plus_inv1 = np.linalg.inv(self.B_plus)
        toc = time.time()
        for i in range(num):        
            B_plus_inv = woodbury_inv(self.B_inv,self.C.T,self.C,self.D)
        toc1 = time.time()
        npt.assert_array_almost_equal(B_plus_inv,B_plus_inv1,decimal = 4)
        self.assertLess(toc1-toc,toc - tic)
        
def main(num=150, num_train=80, num_h=2, tau=3000, total=300, spl_num=5):
    myPara = SampleParams(num, tau, num_h, spl_num, num_train)
    fD = {'f': None, 'c1': None}    
    for k in fD.keys():
        fD[k] = Eval_f()
        fD[k].X,fD[k].X_prd, fD[k].noise_dict = init_x(myPara.X_prd, num_train, h=k)
    
    m, myPara.obj, myPara.cons, icm = setup_model1(fD['f'].X, fD['c1'].X)                
    for k in fD.keys():
        fD[k].mean_prd, fD[k].var_prd = m.predict(fD[k].X_prd, Y_metadata=fD[k].noise_dict)        
    myPara.update(m, fD)  
    
    obj = rosen_constraint(fD['f'].X_prd[:,0:-1])['f'][:,None]
    print(obj-fD['f'].mean_prd)
    print(m.ICM.Mat52.lengthscale)
    print(m.ICM.B.B)
    #check_cov(m, fD, 'c1', [2,2,0])
    return
def setup_model1(X1,X2):
    obj = rosen_constraint(X1)['f'][:,None]
    cons = rosen_constraint(X2)['c1'][:,None]                
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)        
    m['.*Mat52.var'].constrain_fixed(1.)        
    #m['.*Mat52.len'].constrain_fixed(.5)        
    m.optimize() 
    m = m.copy()
    return m,obj,cons,icm   

if __name__ == '__main__':
    main()
    #unittest.main()