# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:01:05 2021

@author: ksdiv
hmm - hmmlearn 
"""

import pickle
import numpy as np
from hmmlearn.hmm import GMMHMM


# encore a modifier, voir ce qui fonctionne le mieux 
# hidden states : Initial contact - loading response - stance - rotation 
# - terminal stance - swing 

# transition matrix - left right , except for rotation

# start prob - (1, 0, 0, ..., 0)

# emission matrix - a trouver av EM algo 
 