# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:30:46 2019
run multi-process in spyder:
Run > Configuration per file > Execute in an external system terminal
This file is the entrance of cKG, also used for run mutiprocess of cKG.py.
@author: 44266
"""
from multiprocessing import Pool
import logging
import sys
import cKG
import os

def main(repeat = 3):    
    import pdb
    pdb.set_trace()
    if len(sys.argv) == 1:
        print("Please input experiment name")
        return
    elif len(sys.argv) > 2:
        print("Too many parameters")
        return
    else:
        EXPname = str(sys.argv[1])
        p = Pool(28)
        for i in range(repeat):
            fname = EXPname + '/output' + str(i)
            if os.path.isdir(fname):
                print("ready to continue experiment " + fname)
                status = 'continue'
            else:
                print("Experiment " + fname + " doesn't exist, make a new one and start")
                os.makedirs(fname)
                status = 'start'
            p.apply_async(run_exp, args=(fname, status, 'off'))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
def run_exp(fname, status, stdout='on'):   
    fname = os.path.abspath(fname)
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    logger.handlers = []
    # Handler
    if status == 'start':
        handler = logging.FileHandler(fname + '/result.log', mode = 'w')
    else:
        handler = logging.FileHandler(fname + '/result.log', mode = 'a')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Handler
    if status == 'start':
        handler = logging.FileHandler(fname +'/result_debug.log', mode = 'w')
    else:
        handler = logging.FileHandler(fname +'/result_debug.log', mode = 'a')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if stdout == 'on':
        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)
    
    cKG.main(num=50, num_train=5, num_h=2, tau=3000, total=300, spl_num=10, 
             num_k=5, fname = fname, status = status, func = "rosen")
def debug_main():
    '''
    For debug use, only run one experiment
    '''
    fname = 'debug_0'
    status = 'start'
    if not os.path.isdir(fname):
        os.makedirs(fname)
        status = 'start'
    run_exp(fname, status, stdout='on')
if __name__ == '__main__':
    debug_main()
    #main()
