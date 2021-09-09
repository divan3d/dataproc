# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:16:34 2021

@author: ksdiv
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
from scipy import interpolate as itp
import pandas as pd
import os

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
file_name = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + "SA" + "/" + "SA" + "_CUT_EGC/shank"     
list_files = os.listdir(file_name)


#%% get error by point 

def add_abs_err(dict_in):
    for key in dict_in:
        dict_in[key]["ae_thigh"] = abs(dict_in[key]["no_mc_thigh_angle"][:] - dict_in[key]["no_mc_kmau_angle"][:])
        dict_in[key]["ae_shank"] = abs(dict_in[key]["no_mc_shank_angle"][:] - dict_in[key]["no_mc_kmal_angle"][:])
    return

def get_equal_time(s_in, length):
    inter = itp.interp1d(range(len(s_in)), s_in)
    out = inter(np.linspace(0, len(s_in)-1, length))
    return out 

def dict_equal_time(dict_in, length):
    eq_d1 = {}
    for key in dict_in:
        temp =  {}
        temp["no_mc_thigh_angle"] = get_equal_time(dict_in[key]["no_mc_thigh_angle"], length)
        temp["no_mc_kmau_angle"] = get_equal_time(dict_in[key]["no_mc_kmau_angle"], length)
        temp["no_mc_shank_angle"] = get_equal_time(dict_in[key]["no_mc_shank_angle"], length)
        temp["no_mc_kmal_angle"] = get_equal_time(dict_in[key]["no_mc_kmal_angle"], length)
        temp["ae_thigh"] = get_equal_time(dict_in[key]["ae_thigh"], length)
        temp["ae_shank"] = get_equal_time(dict_in[key]["ae_shank"], length)
        eq_d1[key] = pd.DataFrame.from_dict(temp)
    return eq_d1

def get_mean(dict_in, name_key):
    res = list(dict_in.keys())[0]  # first key 
    temp_t = dict_in[res][name_key].to_numpy()
    for key in dict_in:
        temp_t = np.dstack((temp_t,dict_in[key][name_key].to_numpy()))
    mean_t = np.mean(temp_t, axis=2)
    return mean_t
    
def plot_sep(dict_in, name_key):
    plt.figure()
    # plt.title("SG thigh and KMAU angles")
    for key in dict_in:
        # plt.plot(dict_in[key][name_key], label = key)
        plt.plot(dict_in[key]["no_mc_kmal_angle"], label = key)
    return


#%%

mean_ae_error = []
mean_shank = []
mean_kmal = []
list_dict = []
for f_name in list_files:
    file_now = op_pickle(file_name + "/" + f_name)
    add_abs_err(file_now)
    temp = dict_equal_time(file_now,150)
    list_dict.append(temp)
    mean_ae_error.append(get_mean(temp, "ae_shank"))
    mean_shank.append(get_mean(temp, "no_mc_shank_angle"))
    mean_kmal.append(get_mean(temp, "no_mc_kmal_angle"))
    

#%% effect speed 

#%%
#FL1
fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
ax1.plot(np.linspace(0,100,150),mean_shank[0][0,:], label = "slow body", color = "lightcoral")
ax1.plot(np.linspace(0,100,150),mean_kmal[0][0,:],label = "slow orthosis", color = "red")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[0][0,:], label = "slow", color = "red")
ax2.set_ylim([0,6])
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
ax2.legend()
fig.suptitle("Mean shank angles & error related to gait cycle, FL1", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[0][0,:], label = "slow body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[0][0,:],label = "slow orthosis", color = "red")
ax1.plot(np.linspace(0,100,150),mean_shank[1][0,:], label = "fast body", color = "lightsteelblue")
ax1.plot(np.linspace(0,100,150),mean_kmal[1][0,:],label = "fast orthosis", color = "blue")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[0][0,:], label = "slow", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[1][0,:], label = "fast", color = "blue")
ax2.set_ylim([0,6])
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
ax2.legend()
fig.suptitle("Mean shank angles & error related to gait cycle, FL1", fontsize = 12)

#%%
#FL3
fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
ax1.plot(np.linspace(0,100,150),mean_shank[2][0,:], label = "slow body", color = "lightcoral")
ax1.plot(np.linspace(0,100,150),mean_kmal[2][0,:], label = "slow orthosis", color = "red")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[2][0,:], label = "slow", color = "red")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.legend()
ax2.grid(axis = "y")
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
ax2.set_ylim([0,8])
fig.suptitle("Mean shank angles & error related to gait cycle, FL3", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[2][0,:], label = "slow body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[2][0,:],label = "slow orthosis", color = "red")
ax1.plot(np.linspace(0,100,150),mean_shank[3][0,:], label = "fast body", color = "lightsteelblue")
ax1.plot(np.linspace(0,100,150),mean_kmal[3][0,:],label = "fast orthosis", color = "blue")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[2][0,:], label = "slow", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[3][0,:], label = "fast", color = "blue")
ax2.set_ylim([0,8])
ax2.legend()
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
fig.suptitle("Mean shank angles & error related to gait cycle, FL3", fontsize = 12)

#%%
#FL5
fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
ax1.plot(np.linspace(0,100,150),mean_shank[4][0,:], label = "slow body", color = "lightcoral")
ax1.plot(np.linspace(0,100,150),mean_kmal[4][0,:], label = "slow orthosis", color = "red")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[4][0,:], label = "slow", color = "red")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.legend()
ax2.grid(axis = "y")
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
ax2.set_ylim([0,14])
fig.suptitle("Mean shank angles & error related to gait cycle, FL5", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[4][0,:], label = "slow body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[4][0,:],label = "slow orthosis", color = "red")
ax1.plot(np.linspace(0,100,150),mean_shank[5][0,:], label = "fast body", color = "lightsteelblue")
ax1.plot(np.linspace(0,100,150),mean_kmal[5][0,:],label = "fast orthosis", color = "blue")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[4][0,:], label = "slow", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[5][0,:], label = "fast", color = "blue")
ax2.set_ylim([0,14])
ax2.legend()
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
fig.suptitle("Mean shank angles & error related to gait cycle, FL5", fontsize = 12)


#%% force 

# slow 
fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
ax1.plot(np.linspace(0,100,150),mean_shank[0][0,:], label = "FL1 body", color = "lightcoral")
ax1.plot(np.linspace(0,100,150),mean_kmal[0][0,:], label = "FL1 orthosis", color = "red")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[0][0,:], label = "FL1", color = "red")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.legend()
ax2.grid(axis = "y")
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
ax2.set_ylim([0,10])
fig.suptitle("Mean shank angles & error related to gait cycle, slow walking", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[0][0,:], label = "FL1 body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[0][0,:],label = "FL1 orthosis", color = "red")
ax1.plot(np.linspace(0,100,150),mean_shank[2][0,:], label = "FL3 body", color = "lightsteelblue")
ax1.plot(np.linspace(0,100,150),mean_kmal[2][0,:],label = "FL3 orthosis", color = "blue")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[0][0,:], label = "FL1", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[2][0,:], label = "FL3", color = "blue")
ax2.set_ylim([0,10])
ax2.legend()
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
fig.suptitle("Mean shank angles & error related to gait cycle, slow walking", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[0][0,:], label = "FL1 body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[0][0,:],label = "FL1 orthosis", color = "red")
# ax1.plot(np.linspace(0,100,150),mean_shank[2][0,:], label = "FL3 body", color = "lightsteelblue")
# ax1.plot(np.linspace(0,100,150),mean_kmal[2][0,:],label = "FL3 orthosis", color = "blue")
ax1.plot(np.linspace(0,100,150),mean_shank[4][0,:], label = "FL5 body", color = "lightgreen")
ax1.plot(np.linspace(0,100,150),mean_kmal[4][0,:],label = "FL5 orthosis", color = "green")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[0][0,:], label = "FL1", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[2][0,:], label = "FL3", color = "blue")
ax2.plot(np.linspace(0,100,150),mean_ae_error[4][0,:], label = "FL5", color = "green")
ax2.set_ylim([0,10])
ax2.legend()
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
fig.suptitle("Mean shank angles & error related to gait cycle, slow walking", fontsize = 12)

#%%
# fast 
fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
ax1.plot(np.linspace(0,100,150),mean_shank[1][0,:], label = "FL1 body", color = "lightcoral")
ax1.plot(np.linspace(0,100,150),mean_kmal[1][0,:], label = "FL1 orthosis", color = "red")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[1][0,:], label = "FL1", color = "red")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.legend()
ax2.grid(axis = "y")
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
ax2.set_ylim([0,14])
fig.suptitle("Mean shank angles & error related to gait cycle, fast walking", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[1][0,:], label = "FL1 body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[1][0,:],label = "FL1 orthosis", color = "red")
ax1.plot(np.linspace(0,100,150),mean_shank[3][0,:], label = "FL3 body", color = "lightsteelblue")
ax1.plot(np.linspace(0,100,150),mean_kmal[3][0,:],label = "FL3 orthosis", color = "blue")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[1][0,:], label = "FL1", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[3][0,:], label = "FL3", color = "blue")
ax2.set_ylim([0,14])
ax2.legend()
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
fig.suptitle("Mean shank angles & error related to gait cycle, fast walking", fontsize = 12)

fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,150),mean_shank[1][0,:], label = "FL1 body", color = "lightcoral")
# ax1.plot(np.linspace(0,100,150),mean_kmal[1][0,:],label = "FL1 orthosis", color = "red")
# ax1.plot(np.linspace(0,100,150),mean_shank[3][0,:], label = "FL3 body", color = "lightsteelblue")
# ax1.plot(np.linspace(0,100,150),mean_kmal[3][0,:],label = "FL3 orthosis", color = "blue")
ax1.plot(np.linspace(0,100,150),mean_shank[5][0,:], label = "FL5 body", color = "lightgreen")
ax1.plot(np.linspace(0,100,150),mean_kmal[5][0,:],label = "FL5 orthosis", color = "green")
ax1.set_xlim([0,100])
ax1.set_ylim([-45,40])
ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
ax2.plot(np.linspace(0,100,150),mean_ae_error[1][0,:], label = "FL1", color = "red")
ax2.plot(np.linspace(0,100,150),mean_ae_error[3][0,:], label = "FL3", color = "blue")
ax2.plot(np.linspace(0,100,150),mean_ae_error[5][0,:], label = "FL5", color = "green")
ax2.set_ylim([0,14])
ax2.legend()
ax2.grid(axis = "y")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("abs error [-]", fontsize = 12)
ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
fig.suptitle("Mean shank angles & error related to gait cycle, fast walking", fontsize = 12)

#%%
# file1 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_CUT_EGC/shank/SA_FL5_2_cut_shank_2.pkl"
# data1 = op_pickle(file1)

# plot_sep(data1, "no_mc_shank_angle")