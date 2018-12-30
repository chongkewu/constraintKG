# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 20:33:40 2018
This script is design for the test without clear value or conclustion, mostly
we use print and plot to observe the result. If we get some deterministic 
conclusion, then it will be postinn the unittest of cKG_test.py,
@author: 44266
"""
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

def main(num=1000, num_train=50, num_h=2, tau=3000, total=300, spl_num=20):
    myPara = SampleParams(num, tau, num_h, spl_num, num_train)
    fD = {'f': None, 'c1': None}    
    for k in fD.keys():
        fD[k] = Eval_f()
        fD[k].X,fD[k].X_prd, fD[k].noise_dict = init_x(myPara.X_prd, num_train, h=k)
    
    m, myPara, icm = setup_model1(fD['f'].X, fD['c1'].X, myPara)                
    for k in fD.keys():
        fD[k].mean_prd, fD[k].var_prd = m.predict(fD[k].X_prd, Y_metadata=fD[k].noise_dict)        
        fD[k].var_prd = fD[k].var_prd.clip(min=0)
    myPara.update(m, fD)  
    spl_set = CRN_gen(fD, 'c1', spl_num)
    print(m)
    print(m.ICM.B.B)    
    check_cov1(m, fD)
    print(m.ICM.Mat52.lengthscale)
    return
    un_star,En_condi = get_un_star1(fD, myPara, m, icm, spl_set)    
    print(fD['f'].mean_prd - En_condi)
    return

def check_cov1(m, fD):
    thr = 0.2
    cov = m.kern.K(fD['f'].X_prd,fD['f'].X_prd)
    cov1 = m.kern.K(fD['f'].X_prd,fD['c1'].X_prd)
    cov2 = m.kern.K(fD['c1'].X_prd,fD['c1'].X_prd)
    ratio = cov2/np.max(cov2)
    
    
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    h_r = ratio.flatten()
    plt.hist(h_r, bins= 50)
    ax1.set_title('Histogram')
    ax1.set_xlabel('cov/max(cov)')
    ax2 = fig.add_subplot(1, 2, 2)
    res = stats.cumfreq(h_r, numbins=100)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,\
                                     res.cumcount.size)
    ax2.bar(x, res.cumcount/len(h_r), width=res.binsize)
    ax2.set_title('Cumulative histogram')
    ax2.set_xlim([x.min(), x.max()])
    ax2.set_xlabel('cov/max(cov)')    
    print(np.min(cov2)/np.max(cov2))
    return
    print(cov.diagonal())      
    return
def setup_model1(X1,X2,myPara):
    obj = rosen_constraint1(X1)['f'][:,None]
    myPara.obj = obj - np.mean(obj)    
    cons = rosen_constraint1(X2)['c1'][:,None]      
    myPara.cons_mean = np.mean(cons)          
    myPara.cons = cons - myPara.cons_mean
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[obj,cons],kernel=icm)        
    m['.*Mat52.var'].constrain_fixed(1.)    
    m['.*Gaussian_noise'].constrain_fixed(.00001)    
    #m['.*Mat52.len'].constrain_fixed(.5)        
    #m.optimize(optimizer = 'scg') 
    m.optimize_restarts(optimizer = 'lbfgsb',num_restarts = 10)
    m = m.copy()
    return m,myPara,icm 
def get_un_star1(fD, myPara, m, icm, spl_set): 
    Pr_feasible = -norm.cdf(0, loc=fD['c1'].mean_prd, scale=np.sqrt(fD['c1'].var_prd))+1
    # get E_n{g(x)|x is feasible}
    un_set = np.zeros((myPara.num, 1))   
    E_n_condi = np.zeros((myPara.num, 1))   
    for j in range(myPara.num):
        obj_pos, count= 0, 0
        spl_x = np.array([myPara.X_prd[j]])
        spl_x1 = add_col(spl_x,0)
        
        for k in range(myPara.spl_num):
            spl_c = spl_set[j][0][k]
            Y = np.vstack([myPara.Y, spl_c])
            
            if spl_c>=-np.inf:# if c>0, update the model with it
                Kx = m.kern.K(myPara.pred_var, spl_x1)
                pred_var = np.vstack([myPara.pred_var, spl_x1])
                K_plus_inv = woodbury_inv(myPara.K_inv, Kx, Kx.T, np.array([[spl_c]]))
                
                mean, var = predict_raw(m, pred_var, spl_x1, Y, K_plus_inv, fullcov=False)
                mean, var = predict_mixed_noise(m, mean, var, full_cov=False, Y_metadata={'output_index':np.array([[0]])})                
                obj_pos += mean
                count += 1
                
        un_set[j] = myPara.tau
        
        if count > 0:
            E_n_condi[j] = obj_pos/count
            un_set[j] = Pr_feasible[j][0] * E_n_condi[j]+myPara.tau * (1-Pr_feasible[j][0])
            
    un_star = min(un_set)
    return un_star, E_n_condi

def rosen_constraint1(params):
  x1 = params[:,0]
  x2 = params[:,1]
  a = 1
  b = 100

  c1 = -x1**2 - (x2-1)**2/2 + 2
  f  = (a - x1)**2 + b*(x2 - x1**2)**2
  return {'f':f, 'c1':400*c1}    

def Mat_Rosen(num=200, num_train = 50):
    X_prd = lhs(2,samples = num)*4-2
    X = X_prd[0:num_train,:]    
    obj = rosen_constraint(X)['f'][:,None]    
    offset_obj = np.mean(obj) 
    obj = obj - offset_obj
    true_obj = rosen_constraint(X_prd)['f'][:,None]    
    
    cons = rosen_constraint(X)['c1'][:,None]  
    cons = cons - np.mean(cons)
    kernel = GPy.kern.Matern52(input_dim=2, ARD = True)
    m = GPy.models.GPRegression(X,obj,kernel,noise_var = 0.01)
    m['.*Gaussian_noise'].constrain_fixed(.000001) 
    #m.optimize(optimizer = 'lbfgsb')
    
    m1 = m.copy()
    m.optimize_restarts(optimizer = 'lbfgsb',num_restarts = 10)
    m1.optimize_restarts(optimizer = 'scg',num_restarts = 2)
    mean,var = m.predict(X_prd)
    mean1,var1 = m1.predict(X_prd)
    dev = mean+offset_obj-true_obj
    dev1 = mean1+offset_obj-true_obj
    print(np.sum(np.abs(dev)))
    print(np.sum(np.abs(dev1)))
    print(m.Mat52.lengthscale)
    print(m1.Mat52.lengthscale)
    print(m._log_marginal_likelihood)
    print(m1._log_marginal_likelihood)


if __name__ == '__main__':
    Mat_Rosen()
    #main()
