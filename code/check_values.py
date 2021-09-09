# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:05:31 2021

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
        
file1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\SA\INTERMEDIATE\SA_FUN_FL5_2.pkl"
data1 = op_pickle(file1)

# file1 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_FL5/SC_FL5_Concentric2.pkl"
# data1 = op_pickle(file1)

# file2 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_TEST_TEST\SA_TEST_TEST_Concentric2.pkl"
# data2 = op_pickle(file2)

# file3 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_TEST_TEST\SA_TEST_TEST_Concentric3.pkl"
# data3 = op_pickle(file3)

#%% check if everything makes sense - especially myosuit vs mocap data 

def plot_Fig1(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["vgrf"] - 500 , label = "vgrf")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.plot(dict_in["t"], dict_in["AlphaShank"] - 80, label = "shank myosuit")
    plt.plot(dict_in["t"], dict_in["R_leg"] * 100, label = "stance R leg")
    plt.plot(dict_in["t"], dict_in["L_leg"] * 100, label = "stance L leg")
    plt.legend()
    plt.title("FIG 1: vgrf, shank angles")
    return 

plot_Fig1(data1)
# plot_Fig1(data2)
# plot_Fig1(data3)

#%%

def plot_Fig2(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["res_norm_thigh"] , label = "res norm thigh")
    plt.plot(dict_in["t"], dict_in["no_mc_thigh_angle"], label = "thigh mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmau_angle"], label = "kmau mocap")
    # plt.plot(dict_in["t"], dict_in["mc_thigh_angle"], label = "thigh mocap")
    # plt.plot(dict_in["t"], dict_in["mc_kmau_angle"], label = "kmau mocap")
    plt.plot(dict_in["t"], dict_in["res_norm_kmau"], label = "res norm kmau")
    plt.plot(dict_in["t"], dict_in["R_leg"] * 50, label = "stance R leg")
    plt.plot(dict_in["t"], dict_in["L_leg"] * 50, label = "stance L leg")
    plt.legend()
    plt.title("FIG 2: mocap, thigh angles, residuals")
    return 

plot_Fig2(data1)
# plot_Fig2(data2)
# plot_Fig2(data3)
#%%

def plot_Fig3(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["res_norm_shank"] , label = "res norm shank")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.plot(dict_in["t"], dict_in["res_norm_kmal"], label = "res norm kmal")
    plt.plot(dict_in["t"], dict_in["R_leg"] * 50, label = "stance R leg")
    plt.plot(dict_in["t"], dict_in["L_leg"] * 50, label = "stance L leg")
    plt.legend()
    plt.title("FIG 3: mocap, shank angles, residuals")
    return 

plot_Fig3(data1)
# plot_Fig3(data2)
# plot_Fig3(data3)

#%%
def plot_Fig2b(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["no_mc_kmau_angle"], label = "kmau mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_thigh_angle"], label = "thigh mocap")
    plt.plot(dict_in["t"], dict_in["AlphaThigh"] - 110, label = "thigh myosuit ")
    plt.plot(dict_in["t"], dict_in["current_sent"], label = "current ")
    # plt.plot(dict_in["t"], dict_in["R_leg"] * 50, label = "stance R leg")
    # plt.plot(dict_in["t"], dict_in["L_leg"] * 50, label = "stance L leg")
    plt.legend()
    plt.grid(True)
    plt.title("FIG 2b: mocap vs myosuit thigh")
    return 

plot_Fig2b(data1)
# plot_Fig2b(data2)
# plot_Fig2b(data3)

#%%

def plot_Fig3b(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["AlphaShank"] -70, label = "shank myosuit")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.plot(dict_in["t"], dict_in["R_leg"] * 30, label = "stance R leg")
    plt.plot(dict_in["t"], dict_in["L_leg"] * 30, label = "stance L leg")
    plt.legend()
    plt.grid(True)
    plt.title("FIG 3b: mocap vs myosuit shank ")
    return 

plot_Fig3b(data1)
# plot_Fig3b(data2)
# plot_Fig3b(data3)

#%%

def plot_Fig4(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["current_sent"], label = "current sent")
    plt.plot(dict_in["t"], dict_in["Force"], label = "Force")
    plt.legend()
    plt.title("FIG 4: current sent and read, force")
    plt.grid(True)
    return 

plot_Fig4(data1)
# plot_Fig4(data2)
# plot_Fig4(data3)

#%%

def plot_Fig5(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["Force"], label = "Force")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    # plt.plot(dict_in["t"], dict_in["AlphaShank"] - 65, label = "shank myosuit")
    plt.legend()
    plt.title("FIG 5:  force, shank angles")
    plt.grid(True)
    return 

plot_Fig5(data1)
# plot_Fig5(data2)
# plot_Fig5(data3)

#%%

def plot_Fig5b(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["vgrf"]/10 - 55 , label = "vgrf")
    plt.plot(dict_in["t"], dict_in["Force"], label = "Force")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    # plt.plot(dict_in["t"], dict_in["AlphaShank"] - 65, label = "shank myosuit")
    plt.legend()
    plt.title("FIG 5b: vgrf  force, shank angles")
    plt.grid(True)
    return 

plot_Fig5b(data1)
# plot_Fig5b(data2)
# plot_Fig5b(data3)

#%%

def plot_Fig6(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["Force"], label = "Force")
    plt.plot(dict_in["t"], dict_in["no_mc_thigh_angle"], label = "thigh mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmau_angle"], label = "kmau mocap")
    # plt.plot(dict_in["t"], dict_in["AlphaShank"] - 65, label = "shank myosuit")
    plt.legend()
    plt.title("FIG 6:  force, thigh angles")
    plt.grid(True)
    return 

plot_Fig6(data1)
# plot_Fig6(data2)
# plot_Fig6(data3)

#%%

def plot_Fig7(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["ForceLevel"], label = "Force Level")
    plt.plot(dict_in["t"], dict_in["Mode"], label = "Mode")
    plt.legend()
    plt.title("FIG 7:  force level and mode")
    plt.grid(True)
    return 

plot_Fig7(data1)
# plot_Fig7(data2)
# plot_Fig7(data3)

#%%

def plot_Fig8(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["GyroCThigh"], label = "gyroscope C thigh")
    plt.plot(dict_in["t"], dict_in["AlphaThigh"] - 115, label = "thigh myosuit")
    plt.plot(dict_in["t"], dict_in["no_mc_thigh_angle"], label = "thigh mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmau_angle"], label = "kmau mocap")
    plt.legend()
    plt.title("FIG 8:  gyro and mocap angles thigh")
    plt.grid(True)
    return 

plot_Fig8(data1)
# plot_Fig8(data2)
# plot_Fig8(data3)

#%%

def plot_Fig9(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["GyroCShank"], label = "gyroscope C shank")
    plt.plot(dict_in["t"], dict_in["AlphaShank"] - 65, label = "shank myosuit")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.legend()
    plt.title("FIG 9:  gyro and mocap angles shank")
    plt.grid(True)
    return 

plot_Fig9(data1)
# plot_Fig9(data2)
# plot_Fig9(data3)

#%%
# faudra changer a accelB ou qqch 
def plot_Fig10(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["AccelCThigh"], label = "accel C thigh")
    # plt.plot(dict_in["t"], dict_in["AlphaThigh"] - 115, label = "thigh myosuit")
    plt.plot(dict_in["t"], dict_in["no_mc_thigh_angle"], label = "thigh mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmau_angle"], label = "kmau mocap")
    plt.legend()
    plt.title("FIG 10:  accel and mocap angles thigh")
    plt.grid(True)
    return 

# plot_Fig10(data1)
# plot_Fig10(data2)
# plot_Fig10(data3)

#%%

def plot_Fig11(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["current_read"] , label = "current read")
    plt.plot(dict_in["t"], dict_in["AccelCShank"], label = "accel C shank")
    # plt.plot(dict_in["t"], dict_in["AlphaShank"] - 65, label = "shank myosuit")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.legend()
    plt.title("FIG 11: accel and mocap angles shank")
    plt.grid(True)
    return 

# plot_Fig11(data1)
# plot_Fig11(data2)
# plot_Fig11(data3)