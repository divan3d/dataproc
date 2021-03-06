# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 15:26:29 2021

@author: ksdiv
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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
# subjects_train = ["SA", "SD", "SG", "SF", "SE", "SH"]#, "SB"]
# SE problematique pr features_dyn_uncut

subjects_train = ["SC", "SB"]

for sub in subjects_train:
    # temp_dir = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + sub + "//" + sub + "_EQ_GC_shank"
    temp_dir = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + sub + "//" + sub + "_features"
    # temp_list = os.listdir(temp_dir)
    temp_list = glob.glob(temp_dir + "/features_fin_thigh*")
    for file in temp_list :
        # whole_dir = temp_dir + "//" + file
        # dict_gc = op_pickle(whole_dir)
        dict_gc = op_pickle(file)
        for key in dict_gc:
            dict_shank[str(counter)] = dict_gc[key]
            counter = counter + 1
    
#%% 

save_obj(dict_shank, "dict_gait_cycle_thigh_only_BC.pkl")

