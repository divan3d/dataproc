# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:01:10 2021

@author: ksdiv
midterm - want to look at error during gait cycle 
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
f_sb_t_1 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SB/SB_CUT_EGC/SB_FL1_cut_thigh_1.pkl"
d_sb_t_1 = op_pickle(f_sb_t_1)

f_sg_t_5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SG/SG_CUT_EGC/SG_FL5_1_cut_thigh_1.pkl"
d_sg_t_5 = op_pickle(f_sg_t_5)

f_sc_t_3 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_CUT_EGC/SC_FL3_cut_thigh_1.pkl"
d_sc_t_3 = op_pickle(f_sc_t_3)

#%%
plt.figure()
plt.plot(d_sb_t_1[33]["no_mc_thigh_angle"])
plt.plot(d_sb_t_1[33]["no_mc_kmau_angle"])
plt.plot(d_sg_t_5[33]["no_mc_thigh_angle"])
plt.plot(d_sg_t_5[33]["no_mc_kmau_angle"])
plt.plot(d_sc_t_3[33]["no_mc_thigh_angle"])
plt.plot(d_sc_t_3[33]["no_mc_kmau_angle"])

#%%

def plot_sep(ex_dict):
    plt.figure()
    for key in ex_dict:
        plt.plot(ex_dict[key]["no_mc_shank_angle"], label = key)
        # plt.plot(ex_dict[key]["no_mc_kmal_angle"], label = key)
    return

plot_sep(d_sg_t_5)


#%% fast dtw 

x = d_sb_t_1[33]["no_mc_thigh_angle"]
y = d_sg_t_5[43]["no_mc_thigh_angle"]

dist, path = fastdtw(x, y, dist = euclidean)

x_idx, y_idx = zip(*path)

x_idx = list(x_idx)
y_idx = list(y_idx)

nx = x[x_idx[:]]
ny = y[y_idx[:]]

nx = nx.reset_index()
ny = ny.reset_index()

#%%
win = scipy.signal.windows.hann(10)

def smooth_hann(ser):
    n0 = ser.iloc[0]
    nE = ser.iloc[-1]
    # n_temp = pd.concat([pd.Series([n0, n0, n0, n0,n0, n0, n0, n0, n0]), ser[:], pd.Series([nE, nE, nE, nE,nE, nE, nE, nE,nE])])
    n_temp = pd.concat([pd.Series([ n0, n0, n0, n0]), ser[:], pd.Series([ nE, nE, nE,nE])])
    n_temp = n_temp.reset_index(drop= True)
        
    n_filt = scipy.signal.convolve(n_temp, win, mode = "same")/sum(win)
    n_fin = n_filt[4:-4]
    return n_fin

nx_filt = smooth_hann(nx["no_mc_thigh_angle"])
ny_filt = smooth_hann(ny["no_mc_thigh_angle"])

#%%
plt.figure()
plt.plot(nx_filt)
plt.plot(ny_filt)
plt.plot(x)
plt.plot(y)
# plt.plot(nx["no_mc_thigh_angle"])
# plt.plot(ny["no_mc_thigh_angle"])
#%% error 
a = pow((d_sg_t_5[43]["no_mc_thigh_angle"][:] - d_sg_t_5[43]["no_mc_kmau_angle"][:]),2)

plt.figure()
plt.plot(d_sg_t_5[43]["no_mc_thigh_angle"])
plt.plot(d_sg_t_5[43]["no_mc_kmau_angle"])
plt.plot(a)
plt.plot(d_sg_t_5[43]["current_sent"])
 