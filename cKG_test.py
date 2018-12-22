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
from numpy.linalg import inv
from GPy.util.linalg import dtrtrs
import time
from scipy.stats import norm

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
        X1,self.X_prd_f,self.noise_dict_f = init_x(X_prd,2,h=0)    
        X2,self.X_prd_c,self.noise_dict_c = init_x(X_prd,2,h=1)
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
        
class Eval_f():
    pass

class SampleParams(object):
    def __init__(self,num,tau,spl_num,num_train):
        self.num = num
        self.tau = tau
        self.spl_num = spl_num
        self.num_train = num_train
        self.X_prd = lhs(2,samples = num)*4-2
    def update(self,m,fD):
        K = m.posterior._K
        self.K_inv = inv(K) 
        self.Y = np.vstack([self.obj,self.cons])
        self.X_prd_all = np.vstack([fD['f'].X_prd,fD['c1'].X_prd])  
        self.pred_var = ObsAr(np.vstack([add_col(fD['f'].X,0),add_col(fD['c1'].X,1)]))

            
def init_x1(X_prd,num_train,h):
    X = X_prd[0:num_train,:]     
    # prediction need to add extra colloum to X_prd to select predicted function 
    if h == 'f':
        X_prd = np.hstack([X_prd,np.zeros_like(X_prd)[:,0][:,None]])
    else:
        X_prd = np.hstack([X_prd,np.ones_like(X_prd)[:,0][:,None]])    
    noise_dict = {'output_index':X_prd[:,2:].astype(int)} 
    return X,X_prd,noise_dict

def main(num=100,num_train=50,num_h=2,tau=3000,total=300,spl_num=10):
    myPara = SampleParams(num,tau,spl_num,num_train)
    fD={'f': None, 'c1': None}
    for k in fD.keys():
        fD[k] = Eval_f()
        fD[k].X,fD[k].X_prd,fD[k].noise_dict = init_x1(myPara.X_prd,num_train,h=k)    
    m,myPara.obj,myPara.cons,icm = setup_model(fD['f'].X,fD['c1'].X)
    for k in fD.keys():
        fD[k].mean_prd,fD[k].var_prd = m.predict(fD[k].X_prd,Y_metadata=fD[k].noise_dict)
    spl_set = m.posterior_samples(fD['c1'].X_prd,size=spl_num,Y_metadata=fD['c1'].noise_dict)
    myPara.update(m,fD)
    tic = time.time()
    get_un_star1(fD,myPara,m,icm,spl_set)
    toc = time.time()
    print('the elapse time:',toc-tic)             
    return  
def get_un_star1(fD,myPara,m,icm,spl_set): 
    Pr_feasible = -norm.cdf(0,loc=fD['c1'].mean_prd,scale=np.sqrt(fD['c1'].var_prd))+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num,1))    
    for j in range(myPara.num):
        obj_pos,count= 0,0
        spl_x = np.array([myPara.X_prd[j]])
        spl_x1 = add_col(spl_x)
        for k in range(myPara.spl_num):
            spl_c = spl_set[j][0][k]
            Y = np.vstack([myPara.Y,spl_c])        
            if spl_c>=0:# if c>0, update the model with it
                Kx = m.kern.K(myPara.pred_var,spl_x1)
                pred_var = np.vstack([myPara.pred_var,spl_x1])
                K_plus_inv = woodbury_inv(myPara.K_inv,Kx,Kx.T,np.array([[spl_c]]))
                
                mean,var = predict_raw(m,pred_var,spl_x1,Y,K_plus_inv,fullcov=False)
                mean,var = predict_mixed_noise(m, mean, var, full_cov=False, Y_metadata={'output_index':np.array([[1]])})                
                obj_pos += mean
                count += 1                
        un_set[j] = myPara.tau
        if count > 0:
            E_n_condi = obj_pos/count
            un_set[j] = Pr_feasible[j][0]*E_n_condi+myPara.tau*(1-Pr_feasible[j][0])
    un_star = min(un_set)
    return un_star 
if __name__ == '__main__':
    main()
    #unittest.main()