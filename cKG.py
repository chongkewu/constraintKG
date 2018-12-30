# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:11:36 2018
This script is developed base on the learn_coregion.py.
We use the cmisoKG to solve the bayesian optimization problem.
Here we deal with the rosenbrock funciton with constraint.
The detail refers to the note miso_with_constraints.

@author: 44266
"""

import GPy
import numpy as np
from pyDOE import *
from scipy.stats import norm
import time
from paramz import ObsAr
from pprint import pprint
from functools import reduce
from numpy.linalg import inv,eig
from copy import copy

def main(num=1000, num_train=5, num_h=2, tau=3000, total=300, spl_num=10):
    myPara = SampleParams(num, tau, num_h, spl_num, num_train)
    fD = {'f': None, 'c1': None}    
    for k in fD.keys():
        fD[k] = Eval_f()
        fD[k].X,fD[k].X_prd, fD[k].noise_dict = init_x(myPara.X_prd, num_train, h=k)
    
    for i in range(total):    
        m, myPara, icm = setup_model(fD['f'].X, fD['c1'].X, myPara)                
        for k in fD.keys():
            fD[k].mean_prd, fD[k].var_prd = m.predict(fD[k].X_prd, Y_metadata=fD[k].noise_dict)        
            fD[k].var_prd = fD[k].var_prd.clip(min=0)
        myPara.update(m, fD)            
        spl_set = CRN_gen(fD, 'c1', spl_num)        
        un_star = get_un_star(fD, myPara, m, icm, spl_set)                 
        En_un_1_set = get_En_un_1_star_fast(fD, myPara, m, icm, num_k=5)
        fD = min_cKG(m, fD, myPara, En_un_1_set, un_star[0])
        
def get_K_inv_set(m, myPara1, fD1):
    dim = myPara1.pred_var.shape[0]+1            
    myPara1.K_plus2_inv = np.zeros((myPara1.num, dim, dim))       
    myPara1.Kx_T = np.zeros((myPara1.num,1,dim))
    for l in range(myPara1.num):   
        spl_x = np.array([fD1['c1'].X_prd[l]])        
        Kxx = m.kern.K(spl_x)
        Kx_T = np.array([fD1['c1'].Kx_T[l]])
        myPara1.K_plus2_inv[l,:,:] = woodbury_inv_check(myPara1.K_inv, Kx_T.T, Kx_T, Kxx)
                
        pred_var = np.vstack([myPara1.pred_var, spl_x])
        myPara1.Kx_T[l,:,:] = m.kern.K(pred_var, spl_x).T
    return myPara1

def get_En_un_1_star_fast(fD, myPara, m, icm, num_k=5):
    En_un_1_set = {'f':None, 'c1':None}
    for ea in fD.keys():        
        En_un_1_set[ea] = Eval_f()
        En_un_1_set[ea].val = np.zeros((myPara.num,1))        
        spl_set_En = CRN_gen(fD, ea, num_k)
        myPara1, fD1 = copy(myPara), copy(fD)
        X_spl = fD[ea].X_prd          
        
        tic = time.time()        
        for ind,spl_ind in enumerate(spl_set_En):                                        
            print(ind)
            print(time.time()-tic)
            tic = time.time()            
            un_1_star_sum,count = 0,0            
            spl_x1 = np.array([X_spl[ind]])                        
            Kx = m.kern.K(myPara.pred_var, spl_x1)            
            Kxx = m.kern.K(spl_x1)
            myPara1.K_inv = woodbury_inv_check(myPara.K_inv, Kx, Kx.T, Kxx)
                                        
            myPara1.pred_var = np.vstack([myPara.pred_var, spl_x1])            
            for ea1 in fD1.keys():
                fD1[ea1].Kx_T = m.kern.K(myPara1.pred_var, fD[ea1].X_prd).T 
                fD1[ea1].var_prd = predict_raw_var(m, myPara1.pred_var,fD[ea1].X_prd, myPara1.K_inv, Y_metadata = fD[ea1].noise_dict)
            myPara1 = get_K_inv_set(m, myPara1, fD1)
            
            for Y_ind in spl_ind[0]:              
                myPara1.Y = np.vstack([myPara.Y, Y_ind])                
                for ea1 in fD1.keys():
                    fD1[ea1].mean_prd = predict_raw_mean(myPara1.K_inv, fD1[ea1].Kx_T, myPara1.Y)                    
                # compute the un+1*
                
                spl_set = CRN_gen(fD1, 'c1', myPara1.spl_num)                    
                un_1_star = get_un_star_fast(fD1, myPara1, m, icm, spl_set)                
                un_1_star_sum += un_1_star
                count += 1     
            # average the un+1* in spl_set                                       
            En_un_1_star = un_1_star_sum/count              
            En_un_1_set[ea].val[ind] = En_un_1_star 
            
    return En_un_1_set

def get_un_star_fast(fD, myPara, m, icm, spl_set):     
    Pr_feasible = -norm.cdf(-myPara.cons_mean, loc=fD['c1'].mean_prd, scale=np.sqrt(fD['c1'].var_prd))+1    
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num, 1)) 
    for j in range(myPara.num):
        obj_pos, count= 0, 0                        
        K_plus_inv = myPara.K_plus2_inv[j,:,:]                       
        Kx_T = myPara.Kx_T[j,:,:]
        for k in range(myPara.spl_num):
            spl_c = spl_set[j][0][k]
            Y = np.vstack([myPara.Y, spl_c])            
            if spl_c >= -myPara.cons_mean:# if c>0, update the model with it        
                #mean, var = predict_raw(m, pred_var, spl_x1, Y, K_plus_inv, fullcov=False)                
                mean = predict_raw_mean(K_plus_inv,Kx_T,Y)                
                obj_pos += mean
                count += 1                
        un_set[j] = myPara.tau        
        if count > 0:
            E_n_condi = obj_pos/count
            un_set[j] = Pr_feasible[j][0] * E_n_condi+myPara.tau * (1-Pr_feasible[j][0])        
    un_star = min(un_set)        
    return un_star

def CRN_gen(fD, task, spl_num):
    # it use common random number replace posterior_samples:
    # spl_set = m.posterior_samples(fD['c1'].X_prd, size=spl_num, Y_metadata=fD['c1'].noise_dict)        
    mean = fD[task].mean_prd
    std = np.sqrt(fD[task].var_prd)                
    rand_norm = np.array([np.random.normal(size = spl_num)])
    spl_crn = np.expand_dims(np.matmul(std,rand_norm)+mean,axis = 1)
    return spl_crn

def check_cov(m, fD, task, x):
    spl_x1 = np.array([x])
    cov = m.kern.K(fD[task].X_prd,spl_x1)
    ss = sum(abs(cov) >= 0.001)
    print(cov)
    print(ss)    
    return

def setup_model(X1,X2,myPara):
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
    m.optimize(optimizer = 'lbfgsb') 
    m = m.copy()
    return m,myPara,icm 

def min_cKG(m, fD, myPara, En_un_1_set, un_star):
    min_cKG = np.inf      
    for ea in fD.keys():
        En_set = En_un_1_set[ea].val - un_star
        min_val = min(En_set)
        ind = np.unravel_index(np.argmin(En_set, axis=None),En_set.shape)        
        if min_val <= min_cKG:            
            min_cKG = min_val
            min_ind = ind
            min_task = ea                            
            print(ea,min_cKG) 
                             
    try:
        spl_pt = np.array([fD[min_task].X_prd[min_ind[0]]])            
    except:        
        raise UnboundLocalError('the cKG computation fails')
    fD[min_task].X = np.vstack([fD[min_task].X,spl_pt[:,0:-1]])

    print(fD['f'].X.shape,fD['c1'].X.shape)
    return fD
     
def get_un_star(fD, myPara, m, icm, spl_set): 
    Pr_feasible = -norm.cdf(-myPara.cons_mean, loc=fD['c1'].mean_prd, scale=np.sqrt(fD['c1'].var_prd))+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num, 1))   

    for j in range(myPara.num):
        obj_pos, count= 0, 0        
        spl_x1 = np.array([fD['c1'].X_prd[j]])
        
        Kx = m.kern.K(myPara.pred_var, spl_x1)        
        kxx = m.kern.K(spl_x1)                
        K_plus_inv = woodbury_inv_check(myPara.K_inv, Kx, Kx.T, kxx)   
        
        pred_var = np.vstack([myPara.pred_var, spl_x1])
        Kx_T = m.kern.K(pred_var, spl_x1).T
        for k in range(myPara.spl_num):
            spl_c = spl_set[j][0][k]
            Y = np.vstack([myPara.Y, spl_c])
            
            if spl_c >= -myPara.cons_mean:# if c>0, update the model with it        
                #mean, var = predict_raw(m, pred_var, spl_x1, Y, K_plus_inv, fullcov=False)                
                mean = predict_raw_mean(K_plus_inv,Kx_T,Y)                
                obj_pos += mean
                count += 1                
        
        un_set[j] = myPara.tau        
        if count > 0:
            E_n_condi = obj_pos/count
            un_set[j] = Pr_feasible[j][0] * E_n_condi+myPara.tau * (1-Pr_feasible[j][0])
        
    un_star = min(un_set)
    return un_star

def get_En_un_1_star(fD, myPara, m, icm, num_k=5):
    En_un_1_set = {'f':None, 'c1':None}
    for ea in fD.keys():        
        En_un_1_set[ea] = Eval_f()
        En_un_1_set[ea].val = np.zeros((myPara.num,1))        
        X_spl = fD[ea].X_prd       
        spl_set_En = CRN_gen(fD, ea, num_k)
        
        myPara1 = copy(myPara)
        fD1 = copy(fD)
        tic = time.time()        
        for ind,spl_ind in enumerate(spl_set_En):                                        
            print(ind)
            print(time.time()-tic)
            tic = time.time()
            
            un_1_star_sum,count = 0,0            
            spl_x1 = np.array([X_spl[ind]])                        
            Kx = m.kern.K(myPara.pred_var, spl_x1)            
            Kxx = m.kern.K(spl_x1)
            myPara1.K_inv = woodbury_inv_check(myPara.K_inv, Kx, Kx.T, Kxx)
                                        
            myPara1.pred_var = np.vstack([myPara.pred_var, spl_x1])
            for ea1 in fD1.keys():
                fD1[ea1].Kx_T = m.kern.K(myPara1.pred_var, fD[ea1].X_prd).T 
                fD1[ea1].var_prd = predict_raw_var(m, myPara1.pred_var,fD[ea1].X_prd, myPara1.K_inv, Y_metadata = fD[ea1].noise_dict)
            for Y_ind in spl_ind[0]:
                myPara1.Y = np.vstack([myPara.Y, Y_ind])                                               
                for ea1 in fD1.keys():
                    fD1[ea1].mean_prd = predict_raw_mean(myPara1.K_inv, fD1[ea1].Kx_T, myPara1.Y)                    
                # compute the un+1*
                
                spl_set = CRN_gen(fD1, 'c1', myPara1.spl_num)
                un_1_star = get_un_star(fD1, myPara1, m, icm, spl_set)
                un_1_star_sum += un_1_star
                count += 1
            # average the un+1* in spl_set                            
            
            En_un_1_star = un_1_star_sum/count              
            En_un_1_set[ea].val[ind] = En_un_1_star   
    
    return En_un_1_set

class Eval_f():
    pass

class SampleParams(object):
    def __init__(self,num,tau,num_h,spl_num,num_train):
        self.num = num
        self.tau = tau
        self.num_h = num_h
        self.spl_num = spl_num
        self.num_train = num_train
        self.X_prd = lhs(2,samples = num)*4-2
    def update(self, m, fD):
        K = m.posterior._K
        try:
            self.K_inv = inv(K)     
        except:
            num = 6
            for i in range(num):
                K = K + 10**(i)*1e-6 * np.eye(K.shape[0])
                try:
                    self.K_inv = inv(K)
                    break
                except:
                    pass                                
            if i == num:
                w,_ = eig(K)
                print('Inverse K is faile, eigenvalue is:',w)        
        self.Y = np.vstack([self.obj,self.cons])
        self.X_prd_all = np.vstack([fD['f'].X_prd,fD['c1'].X_prd])  
        self.pred_var = ObsAr(np.vstack([add_col(fD['f'].X,0),add_col(fD['c1'].X,1)]))
        
def add_col(X,num=0):
    if num == 0:
        X = np.hstack([X,np.zeros_like(X)[:,0][:,None]])
    else:
        X = np.hstack([X,np.ones_like(X)[:,0][:,None]])
    return X
def init_x(X_prd,num_train,h):
    X = X_prd[0:num_train,:]     
    # prediction need to add extra colloum to X_prd to select predicted function 
    if h == 'f':
        X_prd = np.hstack([X_prd,np.zeros_like(X_prd)[:,0][:,None]])
    else:
        X_prd = np.hstack([X_prd,np.ones_like(X_prd)[:,0][:,None]])    
    noise_dict = {'output_index':X_prd[:,2:].astype(int)} 
    return X,X_prd,noise_dict

def predict_raw(m,pred_var,Xnew,Y,K_inv,fullcov=False):
    Kx = m.kern.K(pred_var,Xnew)
    temp = np.matmul(K_inv,Y)
    mu = np.matmul(Kx.T,temp)

    temp1 = np.matmul(K_inv,Kx)    
    if fullcov == True:
        Kxnew = m.kern.K(Xnew)
        var = Kxnew -np.matmul(Kx.T,temp1)
    else:
        Kxnew = m.kern.Kdiag(Xnew)
        var = np.array([Kxnew - np.sum(Kx.T*temp1.T,axis = 1)]).T
    var = var.clip(min=0)
    return mu,var

def predict_raw_var(m, pred_var,Xnew, K_inv, Y_metadata):
    Kxnew = m.kern.Kdiag(Xnew)
    Kx = m.kern.K(pred_var,Xnew)
    temp1 = np.matmul(K_inv,Kx)
    var = np.array([Kxnew - np.sum(Kx.T*temp1.T,axis = 1)]).T
    var = var.clip(min=0)    
    ind = Y_metadata['output_index'].flatten()
    _variance = np.array([m.mixed_noise.likelihoods_list[j].variance for j in ind ])
    var += _variance
    var = var.clip(min=0)
    return var

def predict_raw_mean(K_inv,Kx_T,Y):
    temp = np.matmul(K_inv,Y)
    mu = np.matmul(Kx_T,temp)
    return mu

def predict_mixed_noise(m, mu, var, full_cov=False, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        _variance = np.array([m.mixed_noise.likelihoods_list[j].variance for j in ind ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
        else:
            var += _variance
        return mu, var

def woodbury_inv(P_inv,Q,R,S):
    """
    We use the update formla in Book ML(RW2006) Page 219 equation A.12:
    A =  ||  A_inv = 
    P Q  ||  P1 Q1  
    R S  ||  R1 S1   
    Here S dim is 1 and R = Q.T
    """
    matrixSize = P_inv.shape[0]+1
    A_inv = np.zeros([matrixSize,matrixSize])
    R_P_inv = np.matmul(R,P_inv)
    P_inv_Q = R_P_inv.T
    den = (S - np.matmul(R_P_inv,Q))
    if den == 0:
        return np.nan
    M = 1/den
    A_inv[-1,-1] = M
    A_inv[0:-1,0:-1] = P_inv + reduce(np.matmul,[P_inv_Q,M,R_P_inv])
    tmp = np.matmul(P_inv_Q,M)[:,0]
    A_inv[0:-1,-1] = -tmp
    A_inv[-1,0:-1] = -tmp.T        
    return A_inv

def woodbury_inv_check(P_inv,Q,R,S):
    A_inv = woodbury_inv(P_inv,Q,R,S)
    if np.isnan(A_inv).any():
        num = 6
        for i in range(num):
            jit = 10**(i-6)
            I = np.eye(P_inv.shape[0])
            A_inv = woodbury_inv(P_inv+jit*I, Q, R, S+jit)
            if not np.isnan(A_inv).any():
                break
        if i == num:
            raise ValueError('A_inv do not positive definite even add jitter')
    return A_inv

def rosen_constraint(params):
  x1 = params[:,0]
  x2 = params[:,1]
  a = 1
  b = 100

  c1 = -x1**2 - (x2-1)**2/2 + 2
  f  = (a - x1)**2 + b*(x2 - x1**2)**2
  return {'f':f, 'c1':c1}    
    
if __name__ == '__main__':
    main()