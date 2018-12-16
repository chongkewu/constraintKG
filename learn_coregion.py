# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:30:43 2018
This script generate coregional model of rosenbrock with constraint. It also 
include the sample prediction.
@author: 44266
"""
import GPy
import pylab as pb
import numpy as np
from pyDOE import *

def main():
    num = 10
    num_train = 5
    X_prd = lhs(2,samples = num)*4-2
    X1 = X_prd[0:num_train,:]
    # prediction need to add extra colloum to X_prd to select predicted function 
    X_prd = np.hstack([X_prd,np.zeros_like(X_prd)[:,0][:,None]])
    noise_dict = {'output_index':X_prd[:,2:].astype(int)}

    out = rosen_constraint(X1)
    obj = out['f'][:,None]
    cons = out['c1'][:,None]
    K = GPy.kern.Matern32(2)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X1],[obj,cons],kernel=icm)
    #m['.*Mat32.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
    m.optimize()

    print(m.predict(X_prd,Y_metadata=noise_dict))
   
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