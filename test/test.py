# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:08:50 2018

@author: 44266
"""

import logging
import datetime
from prettytable import PrettyTable

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='example.log',level=logging.DEBUG)

def main():
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning(str(currentDT))

    x = PrettyTable()
    column_names = ["City name", "Area", "Population", "Annual Rainfall"]    
    x.add_column(column_names[0], ["Adelaide", "Brisbane", "Darwin", 
        "Hobart", "Sydney", "Melbourne", "Perth"])
    x.add_column(column_names[1], [1295, 5905, 112, 1357, 2058, 1566, 5386 ])  
    x.add_column(column_names[2], [1158259, 1857594, 120900, 205556, 4336374, 
        3806092, 1554769])  
    x.add_column(column_names[3], [600.5, 1146.4, 1714.7, 619.5, 1214.8, 
        646.9, 869.4])
    logging.info('\n%s',x)
if __name__ == '__main__':
    main()
