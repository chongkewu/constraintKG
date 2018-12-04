# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:08:50 2018

@author: 44266
"""

import logging
import datetime

# =============================================================================
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
# logging.basicConfig(filename='example.log',level=logging.DEBUG)
# 
# =============================================================================
def main():
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning(str(currentDT))

if __name__ == '__main__':
    main()
