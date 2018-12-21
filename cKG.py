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

def main(num=1000,num_train=2,num_h=2,tau=3000,total=300,spl_num=10):
    # initial samples and discrete set A
    X_prd = lhs(2,samples = num)*4-2
    X1,X_prd_f,noise_dict_f = init_x(X_prd,num_train,h=0)    
    X2,X_prd_c,noise_dict_c = init_x(X_prd,num_train,h=1)
    for i in range(total):  
        print(i)      
        m,obj,cons,icm = setup_model(X1,X2)        
        #ss = np.array(m.mixed_noise.likelihoods_list[0].variance)        
        # compute cKG    
        mean_prd_f,var_prd_f = m.predict(X_prd_f,Y_metadata=noise_dict_f)
        mean_prd_c,var_prd_c = m.predict(X_prd_c,Y_metadata=noise_dict_c)        
        x_next = np.array([[2,2,0]])
        #print(m.ICM.K(X_prd_c,x_next))        
        # get samples from c(x), spl_num=100, error=0.1;spl_num=1000;error=0.03      
        spl_set = m.posterior_samples(X_prd_c,size=spl_num,Y_metadata=noise_dict_c)        
        tic = time.time()
        un_star = get_un_star(X1,X2,tau,m,icm,cons,obj,X_prd,X_prd_c,mean_prd_c,\
                              var_prd_c,noise_dict_c,num,spl_num=spl_num,spl_set=spl_set)                
        toc = time.time()        
        print('the total elapse time is',toc - tic)
        return
        # compute En[Un+1*]         
        En_un_1_star_set = get_En_un_1_star(m,num_h,num,tau,X1,X2,obj,cons,\
                                            X_prd,mean_prd_c,var_prd_c,X_prd_f,\
                                            X_prd_c,noise_dict_f,noise_dict_c,icm)       
        # sample and update X1,obj or X2,cons
        cKG_set = En_un_1_star_set-un_star
        min_cKG = np.min(cKG_set)
        ind = np.unravel_index(np.argmin(cKG_set, axis=None),cKG_set.shape)
        h_next,x_next = ind[0],X_prd[ind[1]]        
        print(x_next)
        if h_next == 0:
            X1 = np.vstack([X1,np.array([x_next])])
        else:
            X2 = np.vstack([X2,np.array([x_next])])
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
def get_un_star(X1,X2,tau,m,icm,cons,obj,X_prd,X_prd_c,\
                mean_prd_c,var_prd_c,noise_dict_c,num,spl_num,spl_set):
    Pr_feasible = -norm.cdf(0,loc=mean_prd_c,scale=np.sqrt(var_prd_c))+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((num,1))    
    for j in range(num):
        obj_pos,count= 0,0
        for k in range(spl_num):
            if spl_set[j][0][k]>=0:# if c>0, update the model with it                 
                X2_temp = np.vstack([X2,X_prd[j]])
                cons_temp = np.vstack([cons,spl_set[j][0][k]])          
                m_temp = update_model(m=m,X1=X1,X2=X2_temp,obj=obj,cons=cons_temp,kernel=icm)                
                return
                X_temp = np.array([X_prd[j]])                
                X_prd_tp = np.hstack([X_temp,np.zeros_like(X_temp)[:,0][:,None]])                    
                noise_dict_tp = {'output_index':X_prd_tp[:,2:].astype(int)}  
                mean,var = m_temp.predict(X_prd_tp,Y_metadata=noise_dict_tp)                
                obj_pos += mean
                count += 1                
        un_set[j] = tau
        if count > 0:
            E_n_condi = obj_pos/count
            un_set[j] = Pr_feasible[j][0]*E_n_condi+tau*(1-Pr_feasible[j][0])
    un_star = min(un_set)
    return un_star          
def get_En_un_1_star(m,num_h,num,tau,X1,X2,obj,cons,X_prd,mean_prd_c,\
                     var_prd_c,X_prd_f,X_prd_c,noise_dict_f,noise_dict_c,icm):
    En_un_1_star_set = np.zeros((num_h,num))
    for h_pick in range(num_h):        
        num_k = 2
        if h_pick == 0:
            Y_m,X_spl = noise_dict_f,X_prd_f
        else:
            Y_m,X_spl = noise_dict_c,X_prd_c        
        spl_set_En = m.posterior_samples(X_spl,size=num_k,Y_metadata=Y_m)
        for ind,spl_ind in enumerate(spl_set_En):                
            un_1_star_sum,count = 0,0            
            for Y_ind in spl_ind[0]:
                x_next = X_spl[ind]
                # update the model with x_next and Y_ind
                if h_pick == 0 :
                    X1_temp = np.vstack([X1,np.array([x_next[0:-1]])])                    
                    obj_temp = np.vstack([obj,Y_ind])
                    X2_temp,cons_temp = X2,cons                        
                else:
                    X1_temp,obj_temp = X1,obj
                    X2_temp = np.vstack([X2,np.array([x_next[0:-1]])])                    
                    cons_temp = np.vstack([cons,Y_ind])
                m_temp = update_model(m=m,X1=X1_temp,X2=X2_temp,obj=obj_temp,cons=cons_temp,kernel=icm)                
                # compute the un+1*
                un_1_star = get_un_star(X1_temp,X2_temp,tau,m_temp\
                ,icm,cons_temp,obj_temp,X_prd,X_prd_c,mean_prd_c,\
                var_prd_c,noise_dict_c,num,spl_num = num_k,spl_set=spl_set_En)                
                un_1_star_sum += un_1_star
                count += 1
            # average the un+1* in spl_set 
            En_un_1_star = un_1_star_sum/count
            En_un_1_star_set[h_pick][ind] = En_un_1_star
    return En_un_1_star_set
def update_model(m,X1,X2,obj,cons,kernel):
    v1 = m.mixed_noise.Gaussian_noise_0.variance 
    v2 = m.mixed_noise.Gaussian_noise_1.variance                                
    m_temp = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=kernel)
    m_temp.mixed_noise.Gaussian_noise_0.variance = v1
    m_temp.mixed_noise.Gaussian_noise_1.variance = v2    
    return m_temp
def init_x(X_prd,num_train,h):
    X = X_prd[0:num_train,:]     
    # prediction need to add extra colloum to X_prd to select predicted function 
    if h == 0:
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