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

def main(num=6,num_train=3,num_h=2,tau=3000,total=300):
    # initial samples and discrete set A
    X_prd = lhs(2,samples = num)*4-2
    _,X_prd_f,noise_dict_f = init_x(X_prd,num_train,h=0)
    return
    X2,X_prd_c,noise_dict_c = init_x(X_prd,num_train,h=1)
    for i in range(total):  
        print(i)
        obj = (rosen_constraint(X1)['f'][:,None])
        cons = (rosen_constraint(X2)['c1'][:,None])
        K = GPy.kern.Matern32(2)
        icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
        m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)
        #m['.*Mat32.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
        m.optimize()
        # compute cKG    
        mean_prd_f,var_prd_f = m.predict(X_prd_f,Y_metadata=noise_dict_f)
        mean_prd_c,var_prd_c = m.predict(X_prd_c,Y_metadata=noise_dict_c)

        un_star = get_un_star(X1,X2,tau,m,icm,cons,obj,X_prd,\
          X_prd_c,mean_prd_c,var_prd_c,noise_dict_c,num,spl_num = 10)        

        # compute En[Un+1*] 
        tic = time.time()
        En_un_1_star_set = get_En_un_1_star(m,num_h,\
          num,tau,X1,X2,obj,cons,X_prd,mean_prd_c,var_prd_c,\
          X_prd_f,X_prd_c,noise_dict_f,noise_dict_c,icm)        
        toc = time.time()        
        print('the elapse time is',toc - tic)
        # sample and update X1,obj or X2,cons
        cKG_set = En_un_1_star_set-un_star
        min_cKG = np.min(cKG_set)
        ind = np.unravel_index(np.argmin(cKG_set, axis=None),\
                               cKG_set.shape)
        h_next = ind[0]
        x_next = X_prd[ind[1]]
        print(x_next)
        if h_next == 0:
            X1 = np.vstack([X1,np.array([x_next])])
        else:
            X2 = np.vstack([X2,np.array([x_next])])
        
def get_un_star(X1,X2,tau,m,icm,cons,obj,X_prd,X_prd_c,\
                mean_prd_c,var_prd_c,noise_dict_c,num,spl_num = 10):
     #compute un*
    Pr_feasible = -norm.cdf(0,loc=mean_prd_c,scale=np.sqrt(var_prd_c))+1
        # get E_n{g(x)|x is feasible}
        # get samples from c(x), sql_num=100, error=0.1;sql_num=1000;error=0.03   
    spl_set = m.posterior_samples(X_prd_c,size=spl_num,Y_metadata=noise_dict_c)    
    un_set = np.zeros((num,1))
    for j in range(num):
        obj_pos = 0
        count = 0
        for k in range(spl_num):
            if spl_set[j][0][k]>=0:
            # if c>0, update the model with it 
            # CW: optimize the model here slow the program, 
            # here we use the same parameter. Note: Here seems 
            # only allow one GPCoregionalizedRegression object        
                X2_temp = np.vstack([X2,X_prd[j]])
                cons_temp = np.vstack([cons,spl_set[j][0][k]])     
                m_temp = update_model(m = m,X1=X1,X2=X2_temp,obj=obj,cons=cons_temp,kernel=icm)                
                X_temp = np.array([X_prd[j]])                
                X_prd_tp = np.hstack([X_temp,np.zeros_like(X_temp)[:,0][:,None]])                    
                noise_dict_tp = {'output_index':X_prd_tp[:,2:].astype(int)}                
                mean,var = m_temp.predict(X_prd_tp,Y_metadata=noise_dict_tp)
                obj_pos = obj_pos + mean
                count = count + 1                
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
            Y_m = noise_dict_f
            X_spl = X_prd_f
        else:
            Y_m = noise_dict_c
            X_spl = X_prd_c                
        spl_set = m.posterior_samples(X_spl,size=num_k,Y_metadata=Y_m)
        for ind,spl_ind in enumerate(spl_set):                
            un_1_star_sum = 0
            count = 0
            for Y_ind in spl_ind[0]:
                x_next = X_spl[ind]
                # update the model with x_next and Y_ind
                if h_pick == 0 :
                    X1_temp = np.vstack([X1,np.array([x_next[0:-1]])])
                    X2_temp = X2
                    obj_temp = np.vstack([obj,Y_ind])
                    cons_temp = cons                        
                else:
                    X1_temp = X1
                    X2_temp = np.vstack([X2,np.array([x_next[0:-1]])])
                    obj_temp = obj
                    cons_temp = np.vstack([cons,Y_ind])
                m_temp = update_model(m=m,X1=X1_temp,X2=X2_temp,obj=obj_temp,cons=cons_temp,kernel=icm)                
                # compute the un+1*
                un_1_star = get_un_star(X1_temp,X2_temp,tau,m_temp\
                ,icm,cons_temp,obj_temp,X_prd,X_prd_c,mean_prd_c,\
                var_prd_c,noise_dict_c,num,spl_num = 10)
                un_1_star_sum = un_1_star_sum + un_1_star
                count = count + 1
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