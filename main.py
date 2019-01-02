# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:30:46 2019

@author: 44266
"""

import logging
import sys
import cKG

logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)

logger.handlers = []
# Handler
handler = logging.FileHandler('result.log', mode = 'w')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.INFO)
logger.addHandler(stream_handler)

cKG.main(num=50, num_train=10, num_h=2, tau=3000, total=300, spl_num=10, num_k=5)    