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
import check_poi_shank
import check_poi_shank_entire_gc
import check_poi_thigh_entire_gc

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
subject = "SA"
fl = "FL5_2"

# directory
dir_name_cut = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_CUT_EGC"

#%%
f_1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_" + fl + "\\" + subject + "_" + fl + "_Concentric1.pkl"
d_1 = op_pickle(f_1)

f_2 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_" + fl + "\\" + subject + "_" + fl + "_Concentric2.pkl"
d_2 = op_pickle(f_2)

#%%

# SE_FL3
# d_1 = d_1.iloc[(11700-8839):]
# d_2 = d_2.iloc[(47500-44265):]

# SB_FL3 2 times same leg at beginning 
# d_1 = d_1.iloc[(12130-8800):]

# SF_FL5_1 2 times same leg at beginning 
# d_1 = d_1.iloc[(12180-8513):]

# SG_FL3
# d_2 = d_1.iloc[(60371-50143):]

# SG_FL5_1
# d_1 = d_1.iloc[(12185-8545):]
# d_2 = d_1.iloc[(66329-49704):]

#%% thigh

tc1, ttc1 = check_poi_thigh_entire_gc.keep_good_gc_thigh(d_1)
check_poi_thigh.getMaxDict(tc1)

tc2, ttc2 = check_poi_thigh_entire_gc.keep_good_gc_thigh(d_2)
check_poi_thigh.getMaxDict(tc2)

#%% shank

sc1, stc1 = check_poi_shank_entire_gc.keep_good_gc_shank(d_1)
check_poi_shank.getMaxDict(sc1)

sc2, stc2 = check_poi_shank_entire_gc.keep_good_gc_shank(d_2)
check_poi_shank.getMaxDict(sc2)

#%%

# plt.figure()
# plt.plot(d_1["R_leg"])
# plt.plot(d_1["L_leg"])

# #%%

# plt.figure()
# plt.plot(d_2["R_leg"])
# plt.plot(d_2["L_leg"])
#%%save

os.chdir(dir_name_cut)
name_cut = subject + "_" + fl + "_cut_thigh_1.pkl"
save_obj(tc1, name_cut)


os.chdir(dir_name_cut)
name_cut = subject + "_" + fl + "_cut_thigh_2.pkl"
save_obj(tc2, name_cut)

#%% save shank

os.chdir(dir_name_cut)
name_cut = subject + "_" + fl + "_cut_shank_1.pkl"
save_obj(sc1, name_cut)

os.chdir(dir_name_cut)
name_cut = subject + "_" + fl + "_cut_shank_2.pkl"
save_obj(sc2, name_cut)

#%%

