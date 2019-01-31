# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:30:54 2019

@author: 44266
"""
import time
import pickle
import numpy as np
import numpy.testing as npt
import pandas as pd
from pyDOE import lhs
import GPy
from GPy.util.linalg import pdinv, jitchol
from GPy.util import diag
from prettytable import PrettyTable
from cKG import init_Para_fD, obj_func
from Archive_cKG_woodbury.test import cross_validate, cross_validate_ICM


def cholesky_update_feasibility():
    '''
    test cholesky update feasibility.
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl', 'rb') as f_o:
        _, func_dict, model, ind_j = pickle.load(f_o)
    tic = time.time()
    count = 0
    for ind_j in range(1000):
        cov, _, cov_x, cov_xx = get_cov_plus(model, func_dict, ind_j)
        #cov_plus_chol = jitchol(cov_plus)
        cov_y = cov.copy()
        diag.add(cov_y, 1e-6*np.ones(cov_y.shape[0])+1e-8)
        cov_inv, _, _, _ = pdinv(cov_y)
        #chol_x = np.matmul(chol_inv, cov_x)
        temp = cov_xx - cov_x.T@cov_inv@cov_x
        count = count + 1 if temp > 0 else count
    print("{}/1000 is positive definite".format(count))
    print("elapse time is {} seconds".format(time.time()-tic))
def get_cov_plus(model, func_dict, ind_j):
    '''
    add one column and row to covaraince matrix
    '''
    cov = model.posterior._K
    spl_x1 = func_dict['c1'].X_prd[ind_j][np.newaxis]
    cov_x = model.kern.K(model.X, spl_x1)
    cov_xx = model.kern.K(spl_x1)
    cov_plus = (np.hstack([np.vstack([cov, cov_x.T]), \
                           np.vstack([cov_x, cov_xx])]))
    return cov, cov_plus, cov_x, cov_xx
def cholesky_update_stability():
    '''
    test cholesky update stability.
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl', 'rb') as f_o:
        _, func_dict, model, ind_j = pickle.load(f_o)
    #chol = 0
    count = 0
    tic = time.time()
    for _ in range(10000):
        #chol1 = chol
        cov, _, cov_x, cov_xx = get_cov_plus(model, func_dict, ind_j)
        #cov_plus_chol = jitchol(cov_plus)
        cov_y = cov.copy()
        diag.add(cov_y, 1e-6*np.ones(cov_y.shape[0])+1e-8)
        cov_inv, _, _, _ = pdinv(cov_y)
        #chol_x = np.matmul(chol_inv, cov_x)
        temp = cov_xx - cov_x.T@cov_inv@cov_x
        count = count + 1 if temp > 0 else count
        #print(temp, np.max(abs(chol1 - chol)))
    print("{}/10000 repeatitive sample is postitive definite ".format(count))
    print("elapse time is {} seconds".format(time.time()-tic))
def test_cholesky_update_accuracy():
    '''
    test cholesky update accuracy.
    '''
    fname = 'debug_0'
    with open(fname + '/data_debug_woodbury.pkl', 'rb') as f_o:
        _, func_dict, model, ind_j = pickle.load(f_o)
    cov, cov_plus, cov_x, cov_xx = get_cov_plus(model, func_dict, ind_j)
    cov_plus_chol = jitchol(cov_plus)
    cov_y = cov.copy()
    diag.add(cov_y, 1e-6*np.ones(cov_y.shape[0])+1e-8)
    for _ in range(10):
        cov_inv, chol, chol_inv, _ = pdinv(cov_y)
        chol_x = np.matmul(chol_inv, cov_x)
        temp = cov_xx - cov_x.T@cov_inv@cov_x
        if temp > 0:
            chol_xx = np.sqrt(temp)
            my_chol_plus = np.vstack([np.hstack([chol, chol_x]), \
                                      np.hstack([chol_x.T, chol_xx])])
            print(np.max(abs(my_chol_plus - cov_plus_chol)))
            return
def update_func_d(func_d, func="rosen"):
    '''
    update function dictionary
    '''
    for eac in func_d.keys():
        func_d[eac].obs_val = obj_func(func_d[eac].X, func)[eac][:, None]
        func_d[eac].obs_mean = np.mean(func_d[eac].obs_val)
        func_d[eac].nmlz = func_d[eac].obs_val.std(0)
        func_d[eac].Ny = (func_d[eac].obs_val - func_d[eac].obs_mean)/func_d[eac].nmlz
    return func_d
def test_hyperpara_icm(num_train=80, task = 'f', kfold = 5):
    '''
    Find the theoretical GP icm hyperparameter for function.
    '''
    _, func_d = init_Para_fD(num=1000, tau=3000, num_h=2, spl_num=10, num_train=num_train,\
                         func="rosen", fname="debug_0")
    func_d = update_func_d(func_d)    
    x_in = func_d[task].X
    y_obs = func_d[task].obs_val
    y_norm = func_d[task].Ny    
    mse_icm, lengthscale_icm, r2_icm = cross_validate_ICM(x_in, y_obs, kfold, func_d, k=task)
    print("{} fold cv of {} in GP ICM with y_obs, sample {} points"\
          .format(kfold, task, num_train))
    tab_icm = PrettyTable()
    column_names = ["MSE", "r2_score", "length scale"]
    tab_icm.add_column(column_names[0], mse_icm[:, np.newaxis])
    tab_icm.add_column(column_names[1], r2_icm[:, np.newaxis])
    tab_icm.add_column(column_names[2], lengthscale_icm[:, np.newaxis])
    print(tab_icm)
    print('='*60)
    mse_icm, lengthscale_icm, r2_icm = cross_validate_ICM(x_in, y_norm, kfold, func_d, k=task)
    print("{} fold cv of {} in GP ICM with y_norm, sample {} points"\
          .format(kfold, task, num_train))
    tab_icm = PrettyTable()
    column_names = ["MSE", "r2_score", "length scale"]
    tab_icm.add_column(column_names[0], mse_icm[:, np.newaxis])
    tab_icm.add_column(column_names[1], r2_icm[:, np.newaxis])
    tab_icm.add_column(column_names[2], lengthscale_icm[:, np.newaxis])
    print(tab_icm)
    print('='*60)
def test_hyperpara_gpr(num_train=160, task = 'c1', kfold = 5):
    '''
    Find the theoretical GP regression hyperparameter for function.
    '''

    _, func_d = init_Para_fD(num=1000, tau=3000, num_h=2, spl_num=10, num_train=num_train,\
                         func="rosen", fname="debug_0")
    func_d = update_func_d(func_d)    
    x_in = func_d[task].X
    y_obs = func_d[task].obs_val
    y_norm = func_d[task].Ny
    npt.assert_array_almost_equal(y_norm*func_d[task].nmlz+func_d[task].obs_mean, y_obs)
    mse, lengthscale, r2_sc = cross_validate(x_in, y_obs, kfold)
    print("{} fold cv of {} in GPregression with y_obs, sample {} points"\
          .format(kfold, task, num_train))
    tab_gpr = PrettyTable()
    column_names = ["MSE", "r2_score", "length scale"]
    tab_gpr.add_column(column_names[0], mse[:, np.newaxis])
    tab_gpr.add_column(column_names[1], r2_sc[:, np.newaxis])
    tab_gpr.add_column(column_names[2], lengthscale[:, np.newaxis])
    print(tab_gpr)
    print('='*60)
    mse, lengthscale, r2_sc = cross_validate(x_in, y_norm, kfold)
    print("{} fold cv of {} in GPregression with y_norm, sample {} points"\
          .format(kfold, task, num_train))
    tab_gpr = PrettyTable()
    column_names = ["MSE", "r2_score", "length scale"]
    tab_gpr.add_column(column_names[0], mse[:, np.newaxis])
    tab_gpr.add_column(column_names[1], r2_sc[:, np.newaxis])
    tab_gpr.add_column(column_names[2], lengthscale[:, np.newaxis])
    print(tab_gpr)
    print('='*60)
def test_func_correlation():
    df = pd.DataFrame()
    x_prd = lhs(2,samples=50)*4-2
    df['objective'] = obj_func(x_prd, "rosen")['f']
    df['constraint'] = obj_func(x_prd, "test")['c1']
    print(df.corr())    
    print(df.cov())
    y1 = obj_func(x_prd, "rosen")['f'][:,None]
    y2 = obj_func(x_prd, "rosen")['c1'][:,None]
    x1 = x_prd
    x2 = x_prd
    print(y1.shape,y2.shape,x1.shape,x2.shape)
    
    kern = GPy.kern.Matern52(input_dim=2, ARD = True)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=kern)
    model = GPy.models.GPCoregionalizedRegression([x1,x2],[y1,y2],kernel=icm)
    model['.*Mat52.var'].constrain_fixed(1.)
    model['.*Gaussian_noise'].constrain_fixed(1e-6)  
    model.optimize_restarts(num_restarts=5, verbose=False)
    print(model.kern.B.B)
if __name__ == '__main__':
    #cholesky_update_stability(）
    #cholesky_update_feasibility(）
    #test_cholesky_update_accuracy()
    #test_hyperpara_icm()
    #test_hyperpara_gpr()
    test_func_correlation()