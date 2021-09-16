# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 15:26:29 2021

@author: ksdiv
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
# import scipy.signal

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
dict_shank = {}
counter = 0
subjects_train = ["SA", "SB", "SD", "SE", "SF"]

for sub in subjects_train:
    temp_dir = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + sub + "//" + sub + "_EQ_GC_shank"
    temp_list = os.listdir(temp_dir)
    for file in temp_list :
        whole_dir = temp_dir + "//" + file
        dict_gc = op_pickle(whole_dir)
        dict_shank[str(counter)] = dict_gc
        counter = counter + 1
    
#%% 

save_obj(dict_shank, "shank_gc.pkl")