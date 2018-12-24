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

def main(num=5, num_train=2, num_h=2, tau=3000, total=300, spl_num=5):
    myPara = SampleParams(num, tau, num_h, spl_num, num_train)
    fD = {'f': None, 'c1': None}    
    for k in fD.keys():
        fD[k] = Eval_f()
        fD[k].X,fD[k].X_prd, fD[k].noise_dict = init_x(myPara.X_prd, num_train, h=k)
    for i in range(total):    
        m, myPara.obj, myPara.cons, icm = setup_model(fD['f'].X, fD['c1'].X)                
        for k in fD.keys():
            fD[k].mean_prd, fD[k].var_prd = m.predict(fD[k].X_prd, Y_metadata=fD[k].noise_dict)
        spl_set = m.posterior_samples(fD['c1'].X_prd, size=spl_num, Y_metadata=fD['c1'].noise_dict)
        
        myPara.update(m, fD)
        
        un_star = get_un_star(fD, myPara, m, icm, spl_set)
    
        En_un_1_set = get_En_un_1_star(fD, myPara, m, icm, num_k=5)        
        fD = min_cKG(m, fD, myPara, En_un_1_set)

def check_cov(m,fD):
    spl_x1 = np.array([[2,2,0]])
    cov = m.kern.K(fD['f'].X_prd,spl_x1)
    ss = sum(abs(cov) >= 0.001)
    print(cov)
    print(ss)    
    return

def setup_model(X1,X2):
    obj = rosen_constraint(X1)['f'][:,None]
    cons = rosen_constraint(X2)['c1'][:,None]                
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)        
    m['.*Mat52.var'].constrain_fixed(1.)        
    m.optimize() 
    m = m.copy()
    return m,obj,cons,icm   

def min_cKG(m, fD, myPara, En_un_1_set):
    min_cKG = 2*myPara.tau        
    for ea in fD.keys():
        En_set = En_un_1_set[ea].val
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
    Pr_feasible = -norm.cdf(0, loc=fD['c1'].mean_prd, scale=np.sqrt(fD['c1'].var_prd))+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num, 1))   
    
    for j in range(myPara.num):
        obj_pos, count= 0, 0
        spl_x = np.array([myPara.X_prd[j]])
        spl_x1 = add_col(spl_x,0)
        
        for k in range(myPara.spl_num):
            spl_c = spl_set[j][0][k]
            Y = np.vstack([myPara.Y, spl_c])
            
            if spl_c>=0:# if c>0, update the model with it
                Kx = m.kern.K(myPara.pred_var, spl_x1)
                pred_var = np.vstack([myPara.pred_var, spl_x1])
                K_plus_inv = woodbury_inv(myPara.K_inv, Kx, Kx.T, np.array([[spl_c]]))
                
                mean, var = predict_raw(m, pred_var, spl_x1, Y, K_plus_inv, fullcov=False)
                mean, var = predict_mixed_noise(m, mean, var, full_cov=False, Y_metadata={'output_index':np.array([[0]])})                
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
        Y_m, X_spl = fD[ea].noise_dict, fD[ea].X_prd
        spl_set_En = m.posterior_samples(X_spl,size=num_k,Y_metadata=Y_m)        
        
        myPara1 = copy(myPara)
        myPara1.spl_num = num_k
        fD1 = copy(fD)
        for ind,spl_ind in enumerate(spl_set_En):                
            print(ind)
            un_1_star_sum,count = 0,0            
            spl_x1 = np.array([X_spl[ind]])                        
            Kx = m.kern.K(myPara.pred_var, spl_x1)
            myPara1.pred_var = np.vstack([myPara.pred_var, spl_x1])
            for Y_ind in spl_ind[0]:
                myPara1.Y = np.vstack([myPara.Y, Y_ind])                
                myPara1.K_inv = woodbury_inv(myPara.K_inv, Kx, Kx.T, np.array([[Y_ind]]))                
                for ea1 in fD1.keys():
                    mean, var = predict_raw(m, myPara1.pred_var, fD[ea1].X_prd, myPara1.Y, myPara1.K_inv, fullcov=False)                    
                    fD1[ea1].mean_prd, fD1[ea1].var_prd = predict_mixed_noise(m, mean, var, full_cov=False, Y_metadata=fD[ea1].noise_dict)                    
                # compute the un+1*
                un_1_star = get_un_star(fD1, myPara1, m, icm, spl_set_En)
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
        
def update_model(m,X1,X2,obj,cons,kernel):
    v1 = m.mixed_noise.Gaussian_noise_0.variance 
    v2 = m.mixed_noise.Gaussian_noise_1.variance                                
    m_temp = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=kernel)
    m_temp.mixed_noise.Gaussian_noise_0.variance = v1
    m_temp.mixed_noise.Gaussian_noise_1.variance = v2    
    return m_temp
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
    return mu,var

def predict_mixed_noise(m, mu, var, full_cov=False, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        _variance = np.array([m.mixed_noise.likelihoods_list[j].variance for j in ind ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
        else:
            var += _variance
        return mu, var
def woodbury_inv_origin(P_inv,Q,R,S):
    """
    We use the update formla in Book ML(RW2006) Page 219 equation A.12:
    A =  ||  A_inv = 
    P Q  ||  P1 Q1  
    R S  ||  R1 S1   
    """
    R_P_inv = np.matmul(R,P_inv)
    P_inv_Q = np.matmul(P_inv,Q)
    if S.shape[0] == 1:
        M = 1/(S - np.matmul(R_P_inv,Q))
    else:
        M = np.linalg.inv(S - np.matmul(R_P_inv,Q))
    P1 = P_inv + reduce(np.matmul,[P_inv_Q,M,R_P_inv])
    Q1 = -np.matmul(P_inv_Q,M)
    R1 = -np.matmul(M,R_P_inv)
    S1 = M
    tmp1,tmp2 = map(np.hstack,[[P1,Q1],[R1,S1]])
    A_inv = np.vstack([tmp1,tmp2])
    return A_inv
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
    M = 1/(S - np.matmul(R_P_inv,Q))
    A_inv[-1,-1] = M
    A_inv[0:-1,0:-1] = P_inv + reduce(np.matmul,[P_inv_Q,M,R_P_inv])
    tmp = np.matmul(P_inv_Q,M)[:,0]
    A_inv[0:-1,-1] = -tmp
    A_inv[-1,0:-1] = -tmp.T    
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