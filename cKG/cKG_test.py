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
import matplotlib.pyplot as plt
from scipy import stats

class Test_get_un_star(unittest.TestCase):
    def setUp(self):        
        pass
    @unittest.skip("Need to redesign the test")
    def test_init(self,num=1000, num_train=80, num_h=2, tau=3000, total=300, spl_num=10):        
        myPara, fD = init_Para_fD(num, tau, num_h, spl_num, num_train)
        npt.assert_array_equal(myPara.X_prd[0:num_train,:],fD['c1'].X)
        npt.assert_array_equal(myPara.X_prd[0:num_train,:],fD['f'].X)
        m, myPara, icm = setup_model1(fD['f'].X, fD['c1'].X, myPara)               
        for k in fD.keys():
            fD[k].mean_prd, fD[k].var_prd = m.predict(fD[k].X_prd, Y_metadata=fD[k].noise_dict)        
            fD[k].var_prd = fD[k].var_prd.clip(min=0)
        myPara.update(m, fD)     
        K = m.posterior._K
        Prod = K @ myPara.K_inv
        #print(np.max(np.abs(Prod-np.eye(2*num_train))))
        npt.assert_array_almost_equal(Prod,np.eye(2*num_train),decimal = 0)
        spl_set = CRN_gen(fD, 'c1', spl_num) 
        self.assertEqual(spl_set.shape[0], num)
        self.assertEqual(spl_set.shape[2],spl_num)
        
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
    def test_predict_mean_compare(self):
        self.Xnew = self.X_prd_f
        mean,_ = predict_raw(self.m,self.pred_var,self.Xnew,self.Y,self.K_inv,fullcov=True)
        Kx_T = self.m.kern.K(self.pred_var, self.Xnew).T
        fD = {'f':None}
        fD['f'] = Eval_f(1)
        fD['f'].nmlz = 1
        fD['f'].obs_mean = 0
        mean1 = predict_raw_mean(self.K_inv,Kx_T,self.Y, fD, 'f')       
        npt.assert_array_almost_equal(mean,mean1,decimal = 6)    
    def test_predict_var_compare(self):
# =============================================================================
#         Kx = m.kern.K(self.pred_var, self.X_new)            
#         Kxx = m.kern.K(self.X_new)
#         K_inv = woodbury_inv(self.K_inv, Kx, Kx.T, Kxx) 
# =============================================================================
        
        mean, var = predict_raw(self.m, self.pred_var, self.X_prd_f, self.Y, self.K_inv, fullcov=False)                    
        _, var = predict_mixed_noise(self.m, mean, var, full_cov=False, Y_metadata=self.noise_dict_f)
                
        #pred_var = np.vstack([pred_var, spl_x1])
        var1 = predict_raw_var(self.m, self.pred_var,self.X_prd_f, self.K_inv, Y_metadata = self.noise_dict_f)                    
        npt.assert_array_almost_equal(var,var1,decimal = 6)        
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
def init_x(X_prd, num_train, h):
    X = X_prd[0:num_train,:]     
    # prediction need to add extra colloum to X_prd to select predicted function 
    if h == 'f':
        X_prd = np.hstack([X_prd,np.zeros_like(X_prd)[:,0][:,None]])
    else:
        X_prd = np.hstack([X_prd,np.ones_like(X_prd)[:,0][:,None]])    
    noise_dict = {'output_index':X_prd[:,2:].astype(int)}
    return X, X_prd, noise_dict
def setup_model1(X1,X2,myPara):
    obj = rosen_constraint(X1)['f'][:,None]
    myPara.obj = obj - np.mean(obj)    
    cons = rosen_constraint(X2)['c1'][:,None]      
    myPara.cons_mean = np.mean(cons)          
    myPara.cons = cons - myPara.cons_mean
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)        
    m['.*Mat52.var'].constrain_fixed(1.)    
    m['.*Gaussian_noise'].constrain_fixed(.00001)    
    #m['.*Mat52.len'].constrain_fixed(.5)        
    #m.optimize(optimizer = 'scg') 
    m.optimize_restarts(optimizer = 'lbfgsb',num_restarts = 3)
    m = m.copy()
    return m,myPara,icm 
if __name__ == '__main__':
    unittest.main()