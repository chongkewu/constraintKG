# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:30:43 2018

@author: 44266
"""
import GPy
import pylab as pb
import numpy as np

def main():
    #This functions generate data corresponding to two outputs
    f_output1 = lambda x: 4. * np.cos(x[:,0][:,None]/5.) - 2*x[:,-1][:,None] - 35. + np.random.rand(x.shape[0])[:,None] * 2.
    f_output2 = lambda x: 80. * np.cos(x[:,0][:,None]/20) + 6. * np.cos(x[:,-1][:,None]/5.) + 35. + np.random.rand(x.shape[0])[:,None] * 8.
    
    
    #{X,Y} training set for each output
    X1 = np.random.rand(100,2); X1=X1*75
    X2 = np.random.rand(100,2); X2=X2*70 + 30
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)

    
    K = GPy.kern.Matern32(2)
    icm = GPy.util.multioutput.ICM(input_dim=2,num_outputs=2,kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=icm)
    m['.*Mat32.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
    m.optimize()
    print(m)
    #plot_2outputs(m,Xt1,Yt1,Xt2,Yt2,xlim=(0,100),ylim=(-120,120))

def plot_2outputs(m,Xt1,Yt1,Xt2,Yt2,xlim,ylim):
    fig = pb.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,100),ax=ax1)
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(100,200),ax=ax2)
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)

if __name__ == '__main__':
    main()