# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:20:00 2019
This file include several functions for test and experiment purpose.
@author: 44266
"""
import GPy
from copy import copy, deepcopy
import numpy as np
import pickle
from pyDOE import *
from multiprocessing import Pool
import logging
from paramz import ObsAr
import sys
import cKG
import os
import time
from cKG import obj_func, CRN_gen, predict_raw_mean, woodbury_inv_check,\
 woodbury_inv, setup_model
from GPy.util.linalg import pdinv, dpotrs, dpotri, symmetrify, \
jitchol, dtrtrs, tdot
import matplotlib.pyplot as plt
import numpy.testing as npt
from GPy.util import diag

def read_data(penalty=300,func_name="rosen"):
    '''
    read data from cKG.py
    '''
    with open('cKG_data/01172019/'+func_name+'/output0' + '/data.pkl', 'rb') as f:
        myPara, fD = pickle.load(f)
    coor_ind = myPara.X_ind[:, 0]
    task = myPara.X_ind[:, 1]
    if func_name == "rosen":
        opt_val = 0
    else:
        opt_val = 0.397887
    utl_set = []
    for i, c in enumerate(coor_ind):
        coor = np.array([myPara.X_prd[c, :]])
        #print(coor,task[i])
        if task[i] == 0:
            fea = int(obj_func(coor, func_name)['c1']>=0)
            obj = obj_func(coor, func_name)['f']
            utl = float(obj*fea + (1-fea)*penalty) - opt_val
            utl_set.append(utl)
        else:
            utl_set.append(utl_set[-1])
    print(utl_set)
    fig, ax = plt.subplots()
    ax.plot(range(len(utl_set)), utl_set)
    plt.title("utility of " + func_name)
    plt.xlabel("num of evaluation")
def test_multiprocess():
    '''
    a function for test multiprocesss
    '''
    p = Pool(4)
    for i in range(4):
        p.apply_async(test_run, args=())
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    time.sleep(20)
    return

def test_run():
    print(lhs(2, samples=1000)[0:10, :])
    return
def debug_woodbury_pic():
    '''
    This function plot the distribution of sample points
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl','rb') as f:
        myPara, fD, m, j = pickle.load(f)
    fig, ax = plt.subplots()    
    assert np.unique(fD['f'].X,axis = 0).shape[0] ==  fD['f'].X.shape[0]
    assert np.unique(fD['c1'].X,axis = 0).shape[0] ==  fD['c1'].X.shape[0]    
    ax.scatter(fD['f'].X[:,0],fD['f'].X[:,1])
    ax.scatter(fD['c1'].X[:,0],fD['c1'].X[:,1],marker = 'x')
    ax.legend(('objective', 'constraint'), loc='upper right')
    print("Current determinant of Covariance Matrix: {}".format(np.linalg.det(m.posterior._K)))
    print("Closest two point: {} {}".format(fD['f'].X[20,:],fD['f'].X[2,:]))
    print("Corresponding value: \n{}\n{}".format(obj_func(fD['f'].X[20,:][np.newaxis]),obj_func(fD['f'].X[2,:][np.newaxis])))
    print("So the points is not that closed")
    print("="*60)
    # point  
    print("The hyperparameter is:")
    print(m.ICM.B.B)
    print(m.ICM.Mat52.lengthscale)
    print('='*60)
def debug_model_opt():
    '''
    This function print the hyperparameter of model
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
        myPara, fD, m, j = pickle.load(f)  
    Ny = [fD['f'].Ny,fD['c1'].Ny]
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([fD['f'].X, fD['c1'].X],Ny,kernel=icm)        
    m['.*Mat52.var'].constrain_fixed(1.)    
    m['.*Gaussian_noise'].constrain_fixed(1e-6)         
    m.optimize_restarts(optimizer = 'lbfgsb',num_restarts = 10)
    print(m.ICM.Mat52.lengthscale)
def debug_woodbury_setup(myPara, fD, m, j):
    '''
    a set up funciton for woodbury test
    '''
    # conditional mean     
    spl_c = CRN_gen(fD, 'c1', myPara.spl_num)[j][0][0]    
    spl_x1 = fD['c1'].X_prd[j][np.newaxis]
    spl_x2 = fD['f'].X_prd[j][np.newaxis]
    print("The imaginary sample value of constraint is {}".format(spl_c))    
    Kx = m.kern.K(myPara.pred_var, spl_x1)        
    kxx = m.kern.K(spl_x1)                
    K_plus_inv = woodbury_inv_check(myPara.K_inv, Kx, Kx.T, kxx)           
    pred_var = np.vstack([myPara.pred_var, spl_x1])
    Kx_T = m.kern.K(pred_var, spl_x2).T
    Y = np.vstack([myPara.Y, spl_c])
    K_plus = (np.hstack([\
                    np.vstack([m.posterior._K,Kx.T]),\
                    np.vstack([Kx,kxx])\
                    ])\
    )
    print("The condition number of K_plus: {}".format(np.linalg.cond(K_plus)))                
    

    return K_plus_inv, K_plus, Kx_T, Y , spl_x1, spl_x2, spl_c

def debug_woodbury():
    '''
    Find the difference between m.predict and woodbury updates. 
    When the covariance matrix is singular, the predict performance is poor.
    How to improve it?
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
        myPara, fD, m, j = pickle.load(f)       
    K_plus_inv, K_plus, Kx_T, Y ,spl_x1, spl_x2, spl_c = debug_woodbury_setup\
    (myPara, fD, m, j)

    # find the closest two row of K_plus
    #debug_dist(K_plus,fD)
    print("The point where bug happens is index {} of X_prd".format(j))
    print("The point {} has been sampled at objective function, but haven't at constraint function,\
          the predict mean by GPy model is {}, var is {} "\
          .format(fD['f'].X_prd[j,:][0:2],fD['f'].mean_prd[j],fD['f'].var_prd[j]))
    index = (fD['f'].X == fD['f'].X_prd[j,:][0:2]).all(axis=1).nonzero()    
    print("The sample value is {}".format(fD['f'].obs_val[index[0],:]))
    print("The true value is:{}".format(obj_func(fD['f'].X_prd[j,:][np.newaxis])))
    print('='*60)
    mean = predict_raw_mean(K_plus_inv, Kx_T, Y, fD, 'f')  
    print("The inverse quality use woodbury is {}".format(np.max(K_plus_inv@K_plus - np.eye(K_plus.shape[0]))))
    print("The predict mean use woodbury inverse is {}".format(mean))    
    print('='*60)
    K_plus_inv_np = np.linalg.inv(K_plus)    
    print("The inverse quality use numpy is {}".format(np.max(K_plus_inv_np@K_plus - np.eye(K_plus.shape[0]))))
    mean_np = predict_raw_mean(K_plus_inv_np, Kx_T, Y, fD, 'f')
    print("The predict mean use numpy inverse is {}".format(mean_np)) 
    print('='*60)
    fD['c1'].X = np.vstack([fD['c1'].X,fD['f'].X_prd[j,:][0:2]])    
    m1, _,fD1 = setup_model(fD, "rosen")
    m1.ICM.B.B = m.ICM.B.B
    m1.ICM.Mat52.lengthscale = m.ICM.Mat52.lengthscale        
    mean1,_ = m1.predict(fD1['f'].X_prd, Y_metadata=fD1['f'].noise_dict)        
    fD1['f'].mean_prd = mean1 * fD1['f'].nmlz + fD1['f'].obs_mean
    print("Update model, the m.predict value is{}".format(fD1['f'].mean_prd[j,:]))
    pred_var1 = ObsAr(np.vstack([add_col(fD1['f'].X,0),add_col(fD1['c1'].X,1)]))
    Kx1 = m1.kern.K(pred_var1, spl_x2)         
    Y1 = np.vstack([fD1['f'].Ny,fD1['c1'].Ny])    
    mean_GPy_origin = predict_raw_mean(m1.posterior.woodbury_inv, Kx1.T, Y1, fD1, 'f') 
    print("Update model, use GPy inverse predict_raw_mean is{}".format(mean_GPy_origin))
    mean_cholesky = Kx1.T@m1.posterior.woodbury_vector* fD1['f'].nmlz + fD1['f'].obs_mean
    print("Update model with cholesky decomposition, the mean is{}".format(mean_cholesky))
    print('='*60)
    print("The inverse quality use GPy is {}".format(np.max(m1.posterior.woodbury_inv@K_plus - np.eye(K_plus.shape[0]))))
    mean_GPy = predict_raw_mean(m1.posterior.woodbury_inv, Kx_T, Y, fD, 'f')
    print("The predict mean use GPy model inverse is {}".format(mean_GPy)) 

def add_col( X, num=0):
    '''    
    Add a column for array
    '''
    if num == 0:
        X = np.hstack([X,np.zeros_like(X)[:,0][:,None]])
    else:
        X = np.hstack([X,np.ones_like(X)[:,0][:,None]])
    return X    
def debug_dist(K_plus,fD):
    '''
    measure the distance between the sampled points
    '''
    min_dist,min_ind  = find_min_row(K_plus, method = "square")
    print("The abs maximum and minimum entry in K_plus is {} and {}".format(np.max(np.abs(K_plus)),np.min(np.abs(K_plus))))
    print("min_dist in Covariance maxtrix is {} between row {}".format(min_dist,min_ind))    
    print("corresponding point in constraint {} and {}".format(fD['c1'].X[min_ind[0]-38,:],fD['c1'].X[min_ind[1]-38,:]))
    print("The corresponding true value is:\n {}\n and \n{}".format(\
          obj_func(fD['c1'].X[min_ind[0]-38,:][np.newaxis]),\
          obj_func(fD['c1'].X[min_ind[1]-38,:][np.newaxis])))
    min_dist_f, min_ind_f = find_min_row(fD['f'].X ,method = "square")
    print(min_dist_f, min_ind_f)
    print(fD['f'].X[min_ind_f[0],:],fD['f'].X[min_ind_f[1],:])
    min_dist_c, min_ind_c = find_min_row(fD['c1'].X ,method = "square")
    print(min_dist_c, min_ind_c)
    print(fD['c1'].X[min_ind_c[0],:],fD['c1'].X[min_ind_c[1],:])
    return    

def find_min_row(K_plus, method = "max"):
    '''
    Find the minimum row of array
    '''
    dim = K_plus.shape[0]
    min_dist = np.inf
    min_ind = ()
    for i in range(dim-1):
        for j in range(i+1,dim):
            if method == "max":
                dist=np.max(abs(K_plus[i,:]-K_plus[j,:]))
            else:
                dist = np.sum((K_plus[i,:]-K_plus[j,:])**2)
            if dist < min_dist: 
                min_dist = dist
                min_ind = (i,j)    
    return min_dist,min_ind

def hyperpara_verify():
    '''
    verify the hyperparameter
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
        myPara, fD, m, j = pickle.load(f)        
    X = fD['f'].X
    Y = fD['f'].Ny
    kernel = GPy.kern.Matern52(input_dim = 2)    
    m1 = GPy.models.GPRegression(X,Y,kernel)
    #m1['.*Mat52.var'].constrain_fixed(1.)    
    m1['.*Gaussian_noise'].constrain_fixed(1e-6) 
    m1.optimize_restarts(num_restarts = 10)
    print(m1)
    print('='*60)
    print(m.ICM.Mat52.lengthscale)
def debug_cv():
    '''
    Cross-validation for models.
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
        myPara, fD, m, j = pickle.load(f) 
    kfold = 5
    for k in fD.keys():
        X = fD[k].X
        Y = fD[k].Ny
        mse,lengthscale = cross_validate(X,Y,kfold)
        print("{} fold cross validation of GPregression, The MSE of {} is \n{}"\
              .format(kfold,k,mse[:,np.newaxis]))
        print("{} fold cross validation of GPregression, The lengthscale of {} is \n{}"\
              .format(kfold,k,lengthscale[:,np.newaxis]))
        print('='*60)
        mse_icm,lengthscale_icm = cross_validate_ICM(X,Y,kfold,fD, k=k)
        print("{} fold cross validation of ICM, the MSE of {} is \n{}"\
              .format(kfold,k,mse_icm[:,np.newaxis]))
        print("{} fold cross validation of ICM, the lengthscale of {} is \n{}"\
              .format(kfold,k,lengthscale_icm))
        print('='*60)
def cross_validate(X,Y,kfold):
    '''
    The cross validation of GP model.
    input: X, Y (2D array)
    input: kfold (int)
    return: mse, lengthscale(2D array)
    '''
    mse = np.zeros(kfold)
    lengthscale = np.zeros(kfold)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Input X, Y dimension does not match")
    if X.shape[0] < kfold:
        raise ValueError("kfold larger than X size")
    dim = X.shape[0]
    step = int(dim/kfold)
    for i in range(kfold):
        start = i*step
        end = (i+1)*step if (i!=kfold-1) else None
        X_test = X[start:end,:]
        Y_test = Y[start:end,:]
        X_train = np.vstack([X[0:start,:],X[end:,:]]) if (i!=kfold-1) else X[0:start,:]
        Y_train = np.vstack([Y[0:start,:],Y[end:,:]]) if (i!=kfold-1) else Y[0:start,:]
        #print(X_test.shape,X_train.shape,Y_test.shape,Y_train.shape)        
        kernel = GPy.kern.Matern52(input_dim = 2)    
        m = GPy.models.GPRegression(X_train,Y_train,kernel)
        #m['.*Mat52.var'].constrain_fixed(1.)    
        m['.*Gaussian_noise'].constrain_fixed(1e-6)         
        m.optimize()
        Y_predict,_ = m.predict(X_test)       
        mse[i] = np.sum((Y_test - Y_predict)**2)/Y_test.shape[0]
        lengthscale[i] = float(m.Mat52.lengthscale)        
    return mse, lengthscale
def cross_validate_ICM(X,Y,kfold,fD, k='f'):
    '''
    The cross valiadation for multitask model.
    input: X, Y (2D array)
    input: kfold (int)
    return: mse, lengthscale(2D array)
    Here input fD coming from cKG.py 
    '''
    mse = np.zeros(kfold)
    lengthscale = np.zeros((kfold,2))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Input X, Y dimension does not match")
    if X.shape[0] < kfold:
        raise ValueError("kfold larger than X size")
    dim = X.shape[0]
    step = int(dim/kfold)
    for i in range(kfold):
        start = i*step
        end = (i+1)*step if (i!=kfold-1) else None
        X_test = X[start:end,:]
        Y_test = Y[start:end,:]
        X_train = np.vstack([X[0:start,:],X[end:,:]]) if (i!=kfold-1) else X[0:start,:]
        Y_train = np.vstack([Y[0:start,:],Y[end:,:]]) if (i!=kfold-1) else Y[0:start,:]        
        Ny = [Y_train, fD['c1'].Ny] if k=='f' else [fD['f'].Ny, Y_train]
        K = GPy.kern.Matern52(input_dim=2, ARD = True)
        icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
        if k=='f':
            m = GPy.models.GPCoregionalizedRegression([X_train, fD['c1'].X],Ny,kernel=icm)        
        else:
            m = GPy.models.GPCoregionalizedRegression([fD['f'].X, X_train],Ny,kernel=icm)        
        m['.*Mat52.var'].constrain_fixed(1.)    
        m['.*Gaussian_noise'].constrain_fixed(1e-6)
        m.optimize()   
        dict_noise = copy(fD[k].noise_dict)
        dict_noise['output_index'] = fD[k].noise_dict['output_index'][0:X_test.shape[0],:]
        X_test = add_col(X_test,0) if k=='f' else add_col(X_test,1)
        Y_predict, _ = m.predict(X_test, Y_metadata=dict_noise)
        mse[i] = np.sum((Y_test - Y_predict)**2)/Y_test.shape[0]
        lengthscale[i,:] = np.array([float(m.ICM.Mat52.lengthscale[0]),float(m.ICM.Mat52.lengthscale[1])])
    return mse, lengthscale
def debug_cholesky():
    '''
    Use cholesky to make prediction. How to make the cholesky vector
    (m.posterior.woodbury_vector)?
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl','rb') as f:  
        myPara, fD, m, j = pickle.load(f)       
    K_plus_inv, K_plus, Kx_T, Y ,spl_x1, spl_x2, spl_c = debug_woodbury_setup\
    (myPara, fD, m, j)        
    print("The true value is:{}".format(obj_func(fD['f'].X_prd[j,:][np.newaxis])))
    fD['c1'].X = np.vstack([fD['c1'].X,fD['f'].X_prd[j,:][0:2]])        
    fD1 = deepcopy(fD)    
    for ea in fD1.keys():        
        fD1[ea].obs_val = obj_func(fD1[ea].X, "rosen")[ea][:,None]            
        fD1[ea].Ny = (fD1[ea].obs_val - fD[ea].obs_mean)/fD[ea].nmlz  
    fD1['c1'].Ny[-1,:] = spl_c  
    Ny = [fD1['f'].Ny,fD1['c1'].Ny]
    K = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m1 = GPy.models.GPCoregionalizedRegression([fD['f'].X, fD['c1'].X],Ny,kernel=icm)        
    m1['.*Mat52.var'].constrain_fixed(1.)    
    m1['.*Gaussian_noise'].constrain_fixed(1e-6)         
    m1.optimize(optimizer = 'lbfgsb')       
    m1.ICM.B.B = m.ICM.B.B      
    m1.ICM.Mat52.lengthscale = m.ICM.Mat52.lengthscale 
    mean1,var1 = m1.predict(fD1['f'].X_prd, Y_metadata=fD1['f'].noise_dict)        
    fD1['f'].mean_prd = mean1 * fD1['f'].nmlz + fD1['f'].obs_mean
    print("Update model, the m.predict value is{}".format(fD1['f'].mean_prd[j,:]))
    pred_var1 = ObsAr(np.vstack([add_col(fD1['f'].X,0),add_col(fD1['c1'].X,1)]))
    Kx1 = m1.kern.K(pred_var1, spl_x2)         
    Y1 = np.vstack([fD1['f'].Ny,fD1['c1'].Ny])    
    mean_cholesky = Kx1.T@m1.posterior.woodbury_vector* fD1['f'].nmlz + fD1['f'].obs_mean
    print("Update model with cholesky decomposition, the mean is{}".format(mean_cholesky))    
    print('='*60)    
    Kx1_m = m.kern.K(pred_var1, spl_x2)   
    npt.assert_array_equal(Kx1_m,Kx1)
    print("The L_inf of Y - mean is {}".format(np.max(m.Y - m.posterior.mean)))
    npt.assert_array_equal(K_plus,m1.posterior._K)
    K_chol = jitchol(K_plus)    
    npt.assert_array_equal(m1.posterior.K_chol,K_chol)
    npt.assert_array_equal(Y,Y1)    
    Mywoodbury_vector,_ = dpotrs(K_chol, Y)    
    print("The L_inf of Mywoodbury_vector - woodbury_vector is {}"\
          .format(np.max(Mywoodbury_vector - m1.posterior.woodbury_vector)))          
    Ky = K_plus.copy()
    diag.add(Ky,1e-6*np.ones(Ky.shape[0])+1e-8)
    #Wi, LW, LWi, W_logdet = pdinv(Ky)
    LW = jitchol(Ky)
    alpha, _ = dpotrs(LW, Y, lower=1)    
    print("The L_inf of alpha - woodbury_vector is {}"\
          .format(np.max(alpha- m1.posterior.woodbury_vector)))   
    print("The predict mean with alpha is {}".format(Kx1_m.T@alpha* fD1['f'].nmlz + fD1['f'].obs_mean))
    print('='*60)
    print("The update model predicted variance is {}".format(var1[j,:]))
    Mywoodbury_inv, _ = dpotri(LW, lower=1)
    symmetrify(Mywoodbury_inv)
    print("The L_inf of Mywoodbury_inv - woodbury_inv is {}"\
          .format(np.max(Mywoodbury_inv - m1.posterior.woodbury_inv)))
    Kxx = m.kern.Kdiag(spl_x2)
    tmp = dtrtrs(m1.posterior._woodbury_chol, Kx1_m)[0]
    Myvar = (Kxx - np.square(tmp).sum(0))[:, None]
    print(Myvar+1e-6)
if __name__=='__main__':    
    #read_data()
    #test_multiprocess()
    #debug_woodbury()
    #debug_model_opt()
    #hyperpara_verify()
    #debug_cv()
    debug_cholesky()