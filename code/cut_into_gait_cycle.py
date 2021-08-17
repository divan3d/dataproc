# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 19:42:05 2021

@author: ksdiv

calls functions to cut into gait cycles 
"""

import pickle
import numpy as np
# import math as m
import matplotlib.pyplot as plt
# import scipy.signal
# import statistics
# from scipy import stats
import os
import check_poi_thigh


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
#%% Naming

# change this
subject = "SD"
fl = "FL5"

# directory
dir_name_cut = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_CUT"

#%%
f_in = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_" + fl + "\\" + subject + "_" + fl + "_Concentric1.pkl"
d_in = op_pickle(f_in)

# f_2 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_" + fl + "\\" + subject + "_" + fl + "_Concentric2.pkl"
# d_2 = op_pickle(f_2)
#%%
# plt.figure()
# plt.plot(d_in["no_mc_thigh_angle"])
# plt.plot(d_in["L_leg"])
# plt.plot(d_in["R_leg"])
# plt.plot(d_in["no_mc_kmau_angle"])
# plt.plot(d_in["res_norm_thigh"])
# plt.plot(d_in["res_norm_kmau"])

#%%

# ir = check_poi_thigh.indices_to_cut_R(d_in)
# il = check_poi_thigh.indices_to_cut_L(d_in)

#%%

test, test_list = check_poi_thigh.keep_good_gc_thigh(d_in, 25)
check_poi_thigh.getMaxDict(test)

#%%







