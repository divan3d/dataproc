# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:05:14 2021

@author: ksdiv
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
    
fbs31 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SB/SB_EQ_GC_shank/SB_FL3_cut_shank_1.pkl"
dbs31 = op_pickle(fbs31)

fds52 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SD/SD_EQ_GC_shank/SD_FL5_cut_shank_2.pkl"
dds52 = op_pickle(fds52)

fes12 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SE/SE_EQ_GC_shank/SE_FL1_cut_shank_2.pkl"
des12 = op_pickle(fes12)

fhs32 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_GC_shank/SH_FL3_cut_shank_2.pkl"
dhs32 = op_pickle(fhs32)

fas12 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_shank/SA_FL1_cut_shank_2.pkl"
das12 = op_pickle(fas12)

#%%

def grad_d(d_in):
    """
    gets gradient 

    """ 
    grad = np.gradient(d_in)
    max_grad = max(grad)
    idx_max_grad = np.argmax(grad)
    min_grad = min(grad)
    idx_min_grad = np.argmin(grad)
    return grad, max_grad, idx_max_grad, min_grad, idx_min_grad

def plot_f(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot( - d_in["HallSensor"], label = "HS")
    g, mag, imag, ming, iming = grad_d(-d_in["HallSensor"])
    ax3.plot(g, label = "CS")
    return 

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

plot_f(dbs31[21])
plot_f(dds52[45])
plot_f(des12[12])
plot_f(das12[102])
plot_f(dbs31[64])
plot_f(dds52[8])
plot_f(des12[102])
plot_f(das12[12])

#%%



def plot_g(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot(  d_in["current_read"], label = "HS")
    mincs = min(d_in["current_read"][:110])
    imincs = np.argmin(d_in["current_read"][:110])
    offset = 0.2
    lim_val = mincs + offset
    nbr_pts = sum(d_in["current_read"] <= lim_val)
    # ax2.plot(imincs, mincs, "o")
    vec = np.ones(170)*nbr_pts
    ax3.plot(vec, label = "CS")
    return 

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

plot_g(dbs31[21])
plot_g(dds52[45])
plot_g(des12[12])
plot_g(das12[102])
plot_g(dbs31[64])
plot_g(dds52[8])
plot_g(des12[102])
plot_g(das12[12])

#%% 

def plot_h(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot(d_in["GyroAShank"], label = "GC")
    ax3.plot(d_in["AccelBShank"], label = "AB")
    return 

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

plot_h(dbs31[21])
plot_h(dds52[45])
plot_h(des12[12])
plot_h(das12[102])
plot_h(dbs31[64])
plot_h(dds52[8])
plot_h(des12[102])
plot_h(das12[12])

