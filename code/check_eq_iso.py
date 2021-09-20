# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:15:17 2021

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
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
        
f11 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_iso/SH_FL1_cut_Isometric1.pkl"
d11 = op_pickle(f11) 
f12 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_iso/SH_FL1_cut_Isometric2.pkl"
d12 = op_pickle(f12)

fc32 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_EQ_iso/SC_FL3_cut_Isometric2.pkl"
dc32 = op_pickle(fc32)

t = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/INTERMEDIATE/SH_FUN_FL3.pkl"
tt = op_pickle(t)

d2 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_FL1/SH_FL1_Isometric2.pkl"
f2 = op_pickle(d2) 

ff52 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SF/SF_EQ_iso/SF_FL5_2_cut_Isometric2.pkl"
df52 = op_pickle(ff52)

fa51 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_iso/SA_FL5_2_cut_Isometric1.pkl"
da51 = op_pickle(fa51)
#%%
def plot_a(d_in):
    
    # plt.plot(d_in["GyroCShank"], label = "shank gyro")
    plt.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # plt.plot(d_in["AccelAShank"], label = "shank acc")
    # plt.plot(d_in["GyroCThigh"], label = "thigh gyro")
    # plt.plot(d_in["AccelAThigh"], label = "thigh acc")
    # plt.plot(d_in["Mode"])
    # plt.plot(d_in["current_sent"])
    plt.plot(d_in["Force"])
    plt.legend()
    return


plt.figure()
# plot_a(tt)
plot_a(d11["iso1_on_1"])
plot_a(d11["iso1_on_2"])
plot_a(d12["iso2_on_1"])
plot_a(d12["iso2_on_2"])
plot_a(dc32["iso2_on_1"])
plot_a(dc32["iso2_on_2"])



def plot_b(d_in):
    
    plt.plot(d_in["HallSensor"], label = "HS")
    # plt.plot(d_in["AccelAShank"], label = "shank acc")
    plt.plot(d_in["GyroCThigh"], label = "thigh gyro")
    # plt.plot(d_in["AccelAThigh"], label = "thigh acc")
    # plt.plot(d_in["Mode"])
    # plt.plot(d_in["current_sent"])
    # plt.plot(d_in["Force"])
    plt.legend()
    return


plt.figure()
# plot_a(tt)
plot_b(d11["iso1_on_1"])
plot_b(d11["iso1_on_2"])
plot_b(d12["iso2_on_1"])
plot_b(d12["iso2_on_2"])
plot_b(dc32["iso2_on_1"])
plot_b(dc32["iso2_on_2"])

#%%

dt = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SF/SF_FL5_2/SF_FL5_2_Isometric2.pkl"
f2 = op_pickle(dt)

plt.figure()
plt.plot(f2["no_mc_shank_angle"], label = "shank ang")
plt.plot(f2["Force"], label = "force")

#%%

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

def plot_c(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot(d_in["Force"], label = "force")
    ax3.plot(d_in["current_read"], label = "CS")
    return 

plot_c(d11["iso1_on_1"])
plot_c(d11["iso1_on_2"])
plot_c(d12["iso2_on_1"])
plot_c(d12["iso2_on_2"])
plot_c(dc32["iso2_on_1"])
plot_c(dc32["iso2_on_2"])
    
#%% 

def grad_current(d_in):
    """
    gets gradient 

    """ 
    grad = np.gradient(d_in)
    max_grad = max(grad)
    idx_max_grad = np.argmax(grad)
    min_grad = min(grad)
    idx_min_grad = np.argmin(grad)
    return grad, max_grad, idx_max_grad, min_grad, idx_min_grad

def plot_d(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot(d_in["current_read"], label = "current read")
    minc = d_in["current_read"].min()
    iminc = d_in["current_read"].idxmin()
    ax2.plot(iminc, minc, "o")
    g, mag, imag, mig, imig = grad_current(d_in["current_read"])
    ax3.plot(g)
    ax3.plot(imig, mig, "o")
    return 

#%% check current read

fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1, constrained_layout = True, sharex = True)

def plot_dd(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot(d_in["current_read"], label = "current read")
    minc = d_in["current_read"].min()
    iminc = d_in["current_read"].idxmin()
    # ax2.plot(iminc, minc, "o")
    g, mag, imag, mig, imig = grad_current(d_in["current_read"])
    ax3.plot(g)
    ax3.plot(imig, mig, "o")
    gg, gmag, gimag, gmig, gimig = grad_current(g)
    ax4.plot(gg)
    ax4.plot(gimag, gmag, "o")
    return 

# fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)
fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1, constrained_layout = True, sharex = True)

plot_dd(d11["iso1_on_1"])
plot_dd(d11["iso1_on_2"])
plot_dd(d12["iso2_on_1"])
plot_dd(d12["iso2_on_2"])
plot_dd(dc32["iso2_on_1"])
plot_dd(dc32["iso2_on_2"])
plot_dd(df52["iso2_on_1"])
plot_dd(df52["iso2_on_2"])
plot_dd(da51["iso1_on_1"])
plot_dd(da51["iso1_on_2"])

# fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)
fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1, constrained_layout = True, sharex = True)

plot_dd(d11["iso1_off_1"])
plot_dd(d11["iso1_off_2"])
plot_dd(d12["iso2_off_1"])
plot_dd(d12["iso2_off_2"])
plot_dd(dc32["iso2_off_1"])
plot_dd(dc32["iso2_off_2"])
plot_d(df52["iso2_off_1"])
plot_dd(df52["iso2_off_2"])
plot_dd(da51["iso1_off_1"])
plot_dd(da51["iso1_off_2"])

#%% check gyro 



def width_peak_on(d_in):
    val = sum(d_in["GyroCShank"][:25]>= 5)
    return val
    
def width_peak_off(d_in):
    val = sum(d_in["GyroCShank"][:25]<= -5)
    return val

def plot_e(d_in):
    on_black = np.ones(100)*5
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot(d_in["GyroCShank"], label = "shank gyro")
    ax2.plot(on_black, color = "black", linewidth = 2)
    v= width_peak_on(d_in)
    ax3.plot(np.ones(100)*v)
    # ax3.plot(d_in["GyroCThigh"], label = "shank gyro")
    # ax3.plot(-on_black, color = "black", linewidth = 2)
    return 


fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

plot_e(d11["iso1_on_1"])
plot_e(d11["iso1_on_2"])
plot_e(d12["iso2_on_1"])
plot_e(d12["iso2_on_2"])
plot_e(dc32["iso2_on_1"])
plot_e(dc32["iso2_on_2"])
plot_e(df52["iso2_on_1"])
plot_e(df52["iso2_on_2"])
plot_e(da51["iso1_on_1"])
plot_e(da51["iso1_on_2"])

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

plot_e(d11["iso1_off_1"])
plot_e(d11["iso1_off_2"])
plot_e(d12["iso2_off_1"])
plot_e(d12["iso2_off_2"])
plot_e(dc32["iso2_off_1"])
plot_e(dc32["iso2_off_2"])
plot_e(df52["iso2_off_1"])
plot_e(df52["iso2_off_2"])
plot_e(da51["iso1_off_1"])
plot_e(da51["iso1_off_2"])

#%% need for check encoder on and off 

def get_val_enc(d_in):
    val = - d_in["HallSensor"]
    min_val = val.min()
    min_idx = val.idxmin()
    max_val = val.max()
    max_idx = val.idxmax()
    return min_val, min_idx, max_val, max_idx

def grad_cut(d_in):
    """
    gets gradient 

    """ 
    grad = np.gradient(-d_in["HallSensor"])
    max_grad = max(grad)
    idx_max_grad = np.argmax(grad)
    min_grad = min(grad)
    idx_min_grad = np.argmin(grad)
    return grad, max_grad, idx_max_grad, min_grad, idx_min_grad

#%% check encoder on 
def plot_f(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot( - d_in["HallSensor"], label = "HS")
    min_v, min_i, max_v, max_i = get_val_enc(d_in)
    ax2.plot(min_i, min_v, "o")
    ax2.plot(max_i, max_v, "o")
    g, mg, img, ming, iming = grad_cut(d_in)
    ax3.plot(g, label = "HS der")
    ax3.plot(img,mg, "o")
    return 

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)

# plot_f(d11["iso1_on_1"])
# plot_f(d11["iso1_on_2"])
plot_f(d12["iso2_on_1"])
plot_f(d12["iso2_on_2"])
# plot_f(dc32["iso2_on_1"])
# plot_f(dc32["iso2_on_2"])
# plot_f(df52["iso2_on_1"])
# plot_f(df52["iso2_on_2"])

#%% check encoder off
def plot_g(d_in):
    ax1.plot(d_in["no_mc_shank_angle"], label = "shank ang")
    # ax2.plot(d_in["HallSensor"], label = "HS")
    ax2.plot( - d_in["HallSensor"], label = "HS")
    min_v, min_i, max_v, max_i = get_val_enc(d_in)
    ax2.plot(min_i, min_v, "o")
    ax2.plot(max_i, max_v, "o")
    g, mg, img, ming, iming = grad_cut(d_in)
    ax3.plot(g, label = "HS der")
    ax3.plot(iming,ming, "o")
    return 

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)
# plot_g(d11["iso1_off_1"])
# plot_g(d11["iso1_off_2"])
plot_g(d12["iso2_off_1"])
plot_g(d12["iso2_off_2"])
# plot_g(dc32["iso2_off_1"])
# plot_g(dc32["iso2_off_2"])
# plot_g(df52["iso2_off_1"])
# plot_g(df52["iso2_off_2"])

