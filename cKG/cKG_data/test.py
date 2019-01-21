# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:20:00 2019

@author: 44266
"""
import numpy as np
import pickle
from pyDOE import *
from multiprocessing import Pool
import logging
import sys
import cKG
import os
import time

def read_data():
    with open('exp1/output0' + '/data.pkl','rb') as f:  
        myPara,fD = pickle.load(f)
    print(fD['f'].X)
    with open('exp1/output1' + '/data.pkl','rb') as f:  
        myPara1,fD1 = pickle.load(f)
    print(fD1['f'].X)
    
def test_multiprocess():
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
    
    print(lhs(2,samples=1000)[0:10,:])
    return
if __name__=='__main__':
    #read_data()
    test_multiprocess()
