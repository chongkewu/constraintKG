# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:45:28 2018
When repeated runing experiments, each process generates a .log file. It 
contains all the outputs. When some experiments are broken, remove the 
corresponding .log files and re-run this script.
@author: 44266
"""
import shlex
import subprocess
import os


def main(repeat = 10, repeat_start = 0):
    repeat = int(repeat)
    repeat_start = int(repeat_start)

    for i in range(repeat_start, repeat):
        output_dir = 'output_repeat'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        f_name = 'test'+'_'+str(i)+'.log'
        output_filename = os.path.join(output_dir, f_name)
        if not os.path.isfile(output_filename):
            print('Running experiment %d/%d' % (i+1, repeat))
            output_file = open(output_filename, 'w')
            subprocess.Popen(shlex.split\
                             ("python test.py"),\
                             stdout=output_file, stderr=output_file)





if __name__ == '__main__':
    main()