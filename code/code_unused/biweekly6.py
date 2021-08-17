# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:17:55 2021

@author: ksdiv

get error 
"""


import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats 
from sklearn.metrics import mean_squared_error

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long_features.pkl"
thigh_data = op_pickle(thigh_data_file)

shank_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\shank_good_gc"
shank_data = op_pickle(shank_data_file)

file1 = r"E:\ETHZ\mast_sem_IV\pdm\code02\S01_functions_not_cut.pkl"
data1 = op_pickle(file1)
file3 = r"E:\ETHZ\mast_sem_IV\pdm\code02\S03_functions_not_cut.pkl"
data3 = op_pickle(file3)
file4 = r"E:\ETHZ\mast_sem_IV\pdm\code02\S04_functions_not_cut.pkl"
data4 = op_pickle(file4)
file5 = r"E:\ETHZ\mast_sem_IV\pdm\code02\S05_functions_not_cut.pkl"
data5 = op_pickle(file5)
file6 = r"E:\ETHZ\mast_sem_IV\pdm\code02\S06_functions_not_cut.pkl"
data6 = op_pickle(file6)

file6c = r"E:\ETHZ\mast_sem_IV\pdm\code02\S06_collect.pkl"
data6c = op_pickle(file6c)


#%% 

def get_mse_thigh(d1):
    ms_thigh = mean_squared_error(d1["no_mc_thigh_angle"], d1["no_mc_kmau_angle"])
    tt = d1["no_mc_thigh_angle"] - d1["no_mc_kmau_angle"]
    temp_col = tt.pow(2)
    max_mse = temp_col.max()
    return ms_thigh, max_mse

def get_mse_for_sub(dict_in):
    L = []
    Lm = []
    for key in dict_in:
        a, maxa = get_mse_thigh(dict_in[key])
        L.append(a)
        Lm.append(maxa)
    mean_sub = np.mean(L)
    mean_max_sub = np.mean(Lm)
    return mean_sub, mean_max_sub

def get_mse_all_sub(dict_dict_in):
    LL= []
    LLm = []
    for key in dict_dict_in:
        aa, maxaa = get_mse_for_sub(dict_dict_in[key])
        LL.append(aa)    
        LLm.append(maxaa)
    mean_all = np.mean(LL)
    mean_max_all = np.mean(LLm)
    return mean_all, mean_max_all, LLm

mm, max6 = get_mse_thigh(thigh_data["sub6"][34])
# plot_thigh(thigh_data["sub1"][34])

meanS1, maxs4 = get_mse_for_sub(thigh_data["sub4"])

mean_thigh_good, mean_max_thigh_good, list_max = get_mse_all_sub(thigh_data)

# def get_mse_shank(data):
#     ms = mean_squared_error(data["mc_shank_angle"], data["mc_kmal_angle"])
#     # ms = mean_squared_error(data["no_mc_shank_angle"], data["no_mc_kmal_angle"])
#     return ms 

#%%

def get_mse_shank(d1):
    ms_shank = mean_squared_error(d1["no_mc_shank_angle"], d1["no_mc_kmal_angle"])
    return ms_shank

def get_mse_for_sub_shank(dict_in):
    L = []
    for key in dict_in:
        # a = get_mse_thigh(dict_in[key])
        L.append(get_mse_shank(dict_in[key]))
    mean_sub = np.mean(L)
    return mean_sub

def get_mse_all_sub_shank(dict_dict_in):
    LL= []
    for key in dict_dict_in:
        LL.append(get_mse_for_sub_shank(dict_dict_in[key]))    
    mean_all = np.mean(LL)
    return mean_all

mean_shank_good = get_mse_all_sub_shank(shank_data)

#%%
def plot_sep_thigh(ex_dict,d1):
    """
    plots each gait cycle separetely to form whole data
    - to check that individual gaits were cut correctly 
    
    Parameters
    ----------
    ex_dict : dictionary of dataframe, each containing individual gait cycle

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(d1["t"], d1["no_mc_thigh_angle"], label = "thigh", color = "lightsteelblue")
    plt.plot(d1["t"], d1["no_mc_kmau_angle"], label = "kmau", color = "hotpink")
    # plt.plot(d1["t"], data6c["AlphaThigh"] - 109)
    plt.title("Subject 5 motion capture thigh angle estimations")
    for key in ex_dict:
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_thigh_angle"], color = "blue")
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmau_angle"], color = "red")
        plt.xlabel("time [s]")
        plt.ylabel("angle [deg]")
        # plt.plot(ex_dict[key]["t"], ex_dict[key]["force"]/10, color = "g", label = "force")
        # plt.legend()
    return

plot_sep_thigh(thigh_data["sub5"], data5)

#%%



def plot_sep_shank(ex_dict,d1):
    """
    plots each gait cycle separetely to form whole data
    - to check that individual gaits were cut correctly 
    
    Parameters
    ----------
    ex_dict : dictionary of dataframe, each containing individual gait cycle

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(d1["t"], d1["no_mc_shank_angle"], label = "shank", color = "lightsteelblue")
    plt.plot(d1["t"], d1["no_mc_kmal_angle"], label = "kmal", color = "hotpink")
    plt.title("Subject 5 motion capture angle estimations")
    for key in ex_dict:
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_shank_angle"], color = "blue")
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmal_angle"], color = "red")
        plt.xlabel("time [s]")
        plt.ylabel("angle [deg]")
        # plt.plot(ex_dict[key]["t"], ex_dict[key]["force"]/10, color = "g", label = "force")
        # plt.legend()
    return
# plot_sep_shank(shank_data["sub5"], data5)