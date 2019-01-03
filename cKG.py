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
import logging

def main(num=1000, num_train=10, num_h=2, tau=3000, total=300, spl_num=10, num_k=5):
    np.set_printoptions(linewidth = 150)
    logger = logging.getLogger('main.cKG')
    logger.info('cKG begins at num = %s, num_train = %s, num_h = %s, '
                 'tau = %s, total = %s, spl_num = %s, num_k = %s \n'
                 , num, num_train, num_h, tau, total, spl_num, num_k)    
    myPara, fD = init_Para_fD(num, tau, num_h, spl_num, num_train)
    for i in range(total):    
        m, myPara, icm = setup_model(fD['f'].X, fD['c1'].X, myPara)        
        fD = model_predict(m, fD)
        myPara.update(m, fD) 
        logger.info('Number of sampled points: %s', myPara.pred_var.shape[0])
        un_star = get_un_star(fD, myPara, m, icm, spl_num)                         
        En_un_1_set = get_En_un_1_star_fast(fD, myPara, m, icm, num_k)
        logger.debug('sampled f point and value is\n %s \n', np.hstack([fD['f'].X, myPara.obj * myPara.nmlz_f + myPara.obj_mean]))        
        fD = min_cKG(m, fD, myPara, En_un_1_set, un_star)
        
        

def model_predict(m, fD):
    for k in fD.keys():
        fD[k].mean_prd, fD[k].var_prd = m.predict(fD[k].X_prd, Y_metadata=fD[k].noise_dict)        
        fD[k].var_prd = fD[k].var_prd.clip(min=0)    
    return fD        

def init_Para_fD(num, tau, num_h, spl_num, num_train):
    myPara = SampleParams(num, tau, num_h, spl_num, num_train)
    fD = {'f': None, 'c1': None}    
    for k in fD.keys():
        fD[k] = Func_Dict(myPara.X_prd, num_train, k)        
    return myPara, fD  

def update_fD(m, fD, fD1, myPara1, task):
    for ea1 in fD1.keys():
        if task == 'var':
            fD1[ea1].Kx_T = m.kern.K(myPara1.pred_var, fD[ea1].X_prd).T 
            fD1[ea1].var_prd = predict_raw_var(m, myPara1.pred_var,fD[ea1].X_prd, myPara1.K_inv, Y_metadata = fD[ea1].noise_dict)
        elif task == 'mean':
            fD1[ea1].mean_prd = predict_raw_mean(myPara1.K_inv, fD1[ea1].Kx_T, myPara1.Y)                    
        else:
            raise ValueError('Please input correct task')
    return fD1

def get_En_un_1_star_fast(fD, myPara, m, icm, num_k=5):
    En_un_1_set = {'f':None, 'c1':None}
    for ea in fD.keys():        
        En_un_1_set[ea] = Eval_f(myPara.num)      
        spl_set_En = CRN_gen(fD, ea, num_k)
        myPara1, fD1 = copy(myPara), copy(fD)        
        for ind,spl_ind in enumerate(spl_set_En):                                        
            #print(str(ind)+'/'+str(myPara1.num)+' in task '+ea)                    
            spl_x1 = np.array([fD[ea].X_prd[ind]])              
            myPara1.get_K_inv(m, myPara, spl_x1)
            fD1 = update_fD(m, fD, fD1, myPara1, 'var')
            myPara1.get_K_inv_set(m, fD1)            
            En_un_1_set[ea].val[ind] = Aver_En_un_1_samples(m, icm, fD, fD1, myPara, myPara1, spl_ind[0])          
    logger = logging.getLogger('main.cKG')
    logger.debug("spl_set_En dimension is %s", spl_set_En.shape)
    return En_un_1_set

def Aver_En_un_1_samples(m, icm, fD, fD1, myPara, myPara1, samples):
    un_1_star_sum,count= 0,0
    for Y_ind in samples:              
        myPara1.Y = np.vstack([myPara.Y, Y_ind])    
        fD1 = update_fD(m, fD, fD1, myPara1, 'mean')
        # compute the un+1*
        un_1_star = get_un_star_fast(fD1, myPara1, m, icm)                
        un_1_star_sum += un_1_star
        count += 1     
    # average the un+1* in spl_set                                       
    En_un_1_star = un_1_star_sum/count
    return En_un_1_star

def get_un_star_fast(fD, myPara, m, icm): 
    spl_set = CRN_gen(fD, 'c1', myPara.spl_num)          
    Pr_feasible = -norm.cdf(0, loc=fD['c1'].mean_prd * myPara.nmlz_c1 + myPara.cons_mean, scale=np.sqrt(fD['c1'].var_prd)*myPara.nmlz_c1)+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num, 1)) 
    E_n_condi = np.zeros((myPara.num, 1))
    for j in range(myPara.num):                      
        K_plus_inv = myPara.K_plus2_inv[j,:,:]                       
        Kx_T = myPara.Kx_T[j,:,:]
        un_set,_ = Aver_Un_star_samples(un_set, E_n_condi, j, myPara, spl_set, K_plus_inv, Kx_T, Pr_feasible)
    un_star = np.min(un_set)        
    return un_star

def Aver_Un_star_samples(un_set, E_n_condi, j, myPara, spl_set, K_plus_inv, Kx_T, Pr_feasible):
    obj_pos, count= 0, 0 
    for k in range(myPara.spl_num):
        spl_c = spl_set[j][0][k]
        Y = np.vstack([myPara.Y, spl_c])            
        if spl_c*myPara.nmlz_c1 + myPara.cons_mean  >= 0:# if c>0, update the model with it                        
            mean = predict_raw_mean(K_plus_inv,Kx_T,Y)                
            obj_pos += (mean * myPara.nmlz_f + myPara.obj_mean)
            count += 1                
    un_set[j] = myPara.tau        
    if count > 0:
        E_n_condi[j] = obj_pos/count
        un_set[j] = Pr_feasible[j][0] * E_n_condi[j]+myPara.tau * (1-Pr_feasible[j][0])
    return un_set, E_n_condi


def CRN_gen(fD, task, spl_num):
    # it use common random number replace posterior_samples:
    # spl_set = m.posterior_samples(fD['c1'].X_prd, size=spl_num, Y_metadata=fD['c1'].noise_dict)        
    mean = fD[task].mean_prd
    std = np.sqrt(fD[task].var_prd)                
    rand_norm = np.array([np.random.normal(size = spl_num)])
    spl_crn = np.expand_dims(np.matmul(std,rand_norm)+mean,axis = 1)
    return spl_crn

def setup_model(X1,X2,myPara):
    obj = rosen_constraint(X1)['f'][:,None]
    cons = rosen_constraint(X2)['c1'][:,None]      
    myPara.obj_mean = np.mean(obj)
    myPara.obj = obj - myPara.obj_mean    
    myPara.cons_mean = np.mean(cons)          
    myPara.cons = cons - myPara.cons_mean
    myPara.nmlz_f = np.max(np.abs(myPara.obj))
    myPara.nmlz_c1 = np.max(np.abs(myPara.cons))
    myPara.obj, myPara.cons = [myPara.obj/myPara.nmlz_f,myPara.cons/myPara.nmlz_c1]
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[myPara.obj,myPara.cons],kernel=icm)        
    m['.*Mat52.var'].constrain_fixed(1.)    
    m['.*Gaussian_noise'].constrain_fixed(.00001)     
    try:
        m.optimize(optimizer = 'lbfgsb') 
    except:
        m.optimize(optimizer = 'scg') 
    m = m.copy()
    return m,myPara,icm 

def min_cKG(m, fD, myPara, En_un_1_set, un_star):
    logger = logging.getLogger('main.cKG') 
    min_cKG = np.inf      
    for ea in fD.keys():
        En_set = En_un_1_set[ea].val - un_star
        min_val = min(En_set)
        ind = np.unravel_index(np.argmin(En_set, axis=None),En_set.shape)[0]        
        if min_val <= min_cKG:            
            min_cKG = min_val
            min_ind = ind
            min_task = ea       
    try:
        spl_pt = np.array([fD[min_task].X_prd[min_ind]])            
    except:        
        raise UnboundLocalError('the cKG computation fails')
    fD[min_task].X = np.vstack([fD[min_task].X,spl_pt[:,0:-1]])
    
    logger.info("The Next Sample task is {} at point {} cKG is {}".format(min_task,  str(spl_pt[0,0:-1]), min_cKG[0]))        
    logger.info("The obj value is %s, feasiblility is %s", rosen_constraint(spl_pt)['f'][0], (rosen_constraint(spl_pt)['c1'][0] >0))
    logger.info("The feasible probability is %s", -norm.cdf(-myPara.cons_mean, loc=fD['c1'].mean_prd[min_ind], scale=np.sqrt(fD['c1'].var_prd[min_ind]))+1)
    logger.info("Evaluate f {} times, c1 {} times\n".format(fD['f'].X.shape[0],fD['c1'].X.shape[0]))
    return fD
     
def get_un_star(fD, myPara, m, icm, spl_num): 
    spl_set = CRN_gen(fD, 'c1', spl_num)
    Pr_feasible = -norm.cdf(0, loc=(fD['c1'].mean_prd*myPara.nmlz_c1+myPara.cons_mean), scale=np.sqrt(fD['c1'].var_prd)*myPara.nmlz_c1)+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num, 1))  
    E_n_condi = np.zeros((myPara.num, 1))
    logger = logging.getLogger('main.cKG')
    logger.debug('Pr_feasible dimmension is %s, max %s min %s', Pr_feasible.shape, np.max(Pr_feasible), np.min(Pr_feasible))
    logger.debug('spl_set dimmension is %s', spl_set.shape)
    for j in range(myPara.num):
        spl_x1 = np.array([fD['c1'].X_prd[j]])
        spl_x2 = np.array([fD['f'].X_prd[j]])
        
        Kx = m.kern.K(myPara.pred_var, spl_x1)        
        kxx = m.kern.K(spl_x1)                
        K_plus_inv = woodbury_inv_check(myPara.K_inv, Kx, Kx.T, kxx)   
        
        pred_var = np.vstack([myPara.pred_var, spl_x1])
        Kx_T = m.kern.K(pred_var, spl_x2).T
        un_set, E_n_condi = Aver_Un_star_samples(un_set, E_n_condi, j, myPara, spl_set, K_plus_inv, Kx_T, Pr_feasible)        
    un_star = np.min(un_set)
    ind = np.unravel_index(np.argmin(un_set, axis=None),un_set.shape)[0]
    logger.info("un_star is %s at point %s, expected mean %s, feasible probability %s"
                , un_star, myPara.X_prd[ind,:], E_n_condi[ind,0], Pr_feasible[ind,0])    
    logger.debug("Coordinate (2D), f predict mean, var, un_set, E_n_condi, feasiblity and f predict mean use K_inv  is \n%s\n"
                 , np.hstack([myPara.X_prd, myPara.nmlz_f*fD['f'].mean_prd + myPara.obj_mean, myPara.nmlz_f**2*fD['f'].var_prd, un_set, E_n_condi, Pr_feasible,
                              myPara.nmlz_f*predict_raw_mean(myPara.K_inv, m.kern.K(myPara.pred_var, fD['f'].X_prd).T , myPara.Y) + myPara.obj_mean])[0:50,:])    
    logger.debug("K_inv, myPara.Y and pred_var dimension is %s", [myPara.K_inv.shape, myPara.Y.shape, myPara.pred_var.shape])    
    logger.debug("K max and min is %s", [np.max(m.posterior._K),np.min(m.posterior._K)])
    return un_star


class Func_Dict(object):
    def __init__(self, X_prd, num_train, task):
        self.X = X_prd[0:num_train,:]     
        # prediction need to add extra colloum to X_prd to select predicted function 
        if task == 'f':
            self.X_prd = np.hstack([X_prd,np.zeros_like(X_prd)[:,0][:,None]])
        else:
            self.X_prd = np.hstack([X_prd,np.ones_like(X_prd)[:,0][:,None]])    
        self.noise_dict = {'output_index':self.X_prd[:,2:].astype(int)}

class Eval_f(object):
    def __init__(self, num):
        self.val = np.zeros((num,1))      

class SampleParams(object):
    def __init__(self,num,tau,num_h,spl_num,num_train):
        self.num = num
        self.tau = tau
        self.num_h = num_h
        self.spl_num = spl_num
        self.num_train = num_train
        self.X_prd = lhs(2,samples = num)*4-2
    def update(self, m, fD, num = 6, thres = 1e-6):
        K = m.posterior._K
        coef = np.max(np.abs(K))
        K = K/coef
        w,_ = eig(K)
        if min(w) > thres:
            self.K_inv = inv(K)     
        else:
            for i in range(num):
                K = K + 10**(i)*1e-6 * np.eye(K.shape[0])                
                w,_ = eig(K)
                if min(w) > thres:
                    break
            if i == num - 1:
                w,_ = eig(K)
                raise ValueError('Inverse K is faile, eigenvalue is:{}'.format(w))        
            else:
                self.K_inv = inv(K) 
        self.K_inv = self.K_inv/coef
        K = K*coef
        logger = logging.getLogger('main.cKG')            
        logger.debug("robust K_inv difference is: %s", np.max(np.abs(np.matmul(self.K_inv, m.posterior._K)-np.eye(self.K_inv.shape[0]))))        
        self.Y = np.vstack([self.obj,self.cons])
        self.pred_var = ObsAr(np.vstack([self.add_col(fD['f'].X,0),self.add_col(fD['c1'].X,1)]))
    def get_K_inv(self, m, myPara, spl_x1):
        Kx = m.kern.K(myPara.pred_var, spl_x1)            
        Kxx = m.kern.K(spl_x1)
        self.K_inv = woodbury_inv_check(myPara.K_inv, Kx, Kx.T, Kxx)                            
        self.pred_var = np.vstack([myPara.pred_var, spl_x1])         
    def get_K_inv_set(self, m, fD1):
        dim = self.pred_var.shape[0]+1            
        self.K_plus2_inv = np.zeros((self.num, dim, dim))       
        self.Kx_T = np.zeros((self.num,1,dim))
        for l in range(self.num):   
            spl_x = np.array([fD1['c1'].X_prd[l]])     
            spl_x1 = np.array([fD1['f'].X_prd[l]]) 
            Kxx = m.kern.K(spl_x)
            Kx_T = np.array([fD1['c1'].Kx_T[l]])
            self.K_plus2_inv[l,:,:] = woodbury_inv_check(self.K_inv, Kx_T.T, Kx_T, Kxx)
                    
            pred_var = np.vstack([self.pred_var, spl_x])
            self.Kx_T[l,:,:] = m.kern.K(pred_var, spl_x1).T            
    def add_col(self, X, num=0):
        if num == 0:
            X = np.hstack([X,np.zeros_like(X)[:,0][:,None]])
        else:
            X = np.hstack([X,np.ones_like(X)[:,0][:,None]])
        return X

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