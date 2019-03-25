# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:50:43 2019
This script a set of draft code.
@author: Chongke Wu
"""
import numpy as np
import GPy
from MLP_obj_c import run_obj, run_cons
from  pyDOE import lhs

def fit_11_dim_GP_icm():
    '''
    this function fit the GP icm with 10 dimension input.
    '''
    obj_x = lhs(10, samples=5)
    cons_x = lhs(10, samples=4)
    obj_y = np.random.rand(5, 1)
    cons_y = np.random.rand(4,1)
    Ny = [obj_y, cons_y]
    K = GPy.kern.Matern52(input_dim=10, ARD=True)
    icm = GPy.util.multioutput.ICM(input_dim=10, num_outputs=2, kernel=K)
    m = GPy.models.GPCoregionalizedRegression([obj_x, cons_x], Ny, kernel=icm)
    m['.*Mat52.var'].constrain_fixed(1.)
    m['.*Gaussian_noise'].constrain_fixed(1e-6)
    m.optimize()
    #print(m)
    #print(m.ICM.Mat52.lengthscale)
    x_prd = np.random.rand(5, 10)
    noise_dict = {'output_index': x_prd[:, -1].astype(int)}
    x_prd = add_col(x_prd, num=0)    
    print(m.predict(x_prd, Y_metadata=noise_dict))
    print('obj_x is {}'.format(obj_x))
    return
def add_col(X, num=0):
    if num == 0:
        X = np.hstack([X, np.zeros_like(X)[:, 0][:, None]])
    else:
        X = np.hstack([X, np.ones_like(X)[:, 0][:, None]])
    return X
def run_MLP():
    '''
    this function try to run MLP_obj_c.py as objective and constraint function.    
    '''
    out1 = run_obj()
    out2 = run_cons()
    print(out1, out2)
    return
    
if __name__ == "__main__":
    fit_11_dim_GP_icm()        
    #run_MLP()