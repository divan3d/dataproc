# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:33:29 2021


compare erreur entre sujet et à travers meme sujet pr thigh et shank,
pr les valeurs qui restent après nettoyage 

structure : dict contenant chaque sujet -> dict contenant gait cycle -> dataframe 
@author: ksdiv
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
from sklearn.metrics import mean_absolute_error
import scipy.signal 

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
#%% import files 

file_shank =   r"E:\ETHZ\mast_sem_IV\pdm\code\shank_good_gc"
data_shank = op_pickle(file_shank)
file_thigh =   r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc"
data_thigh = op_pickle(file_thigh)
  
#%%      
file1 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S01_Mocap_angles_removed_offset_res_MS.pkl"
data1 = op_pickle(file1)

#%%  

def mae_error_ind_gc_shank(data_in):
    return mean_absolute_error(data_in["no_mc_kmal_angle"], data_in["no_mc_shank_angle"])

def mae_error_ind_gc_thigh(data_in):
    return mean_absolute_error(data_in["no_mc_kmau_angle"], data_in["no_mc_thigh_angle"])

def mae_one_sub_shank(dict_in):
    dict_mae = {}
    counter = 0
    for key in dict_in:
        dict_mae[counter] = mae_error_ind_gc_shank(dict_in[key])
        counter = counter + 1
    return dict_mae

def mae_one_sub_thigh(dict_in):
    dict_mae = {}
    counter = 0
    for key in dict_in:
        dict_mae[counter] = mae_error_ind_gc_thigh(dict_in[key])
        counter = counter + 1
    return dict_mae
        
#%%

# sub1_shank = mae_one_sub_shank(data_shank["sub1"])

# sub1_thigh = mae_one_sub_thigh(data_thigh["sub1"])

#%% plot

def plot_curves(dict_in):
    plt.figure()
    for key in dict_in:
        plt.plot(range(len(dict_in[key])), dict_in[key]["no_mc_shank_angle"])
    return 

# plot_curves(data_shank["sub1"])

#%%

def plot_curves_thigh(dict_in):
    plt.figure()
    for key in dict_in:
        plt.plot(range(len(dict_in[key])), dict_in[key]["no_mc_thigh_angle"])
    return 


# plot_curves_thigh(data_thigh["sub1"])

#%%

def plot_sep_shank(ex_dict):
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
    # plt.figure()
    for key in ex_dict:
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_shank_angle"], color = "b")
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmal_angle"], color = "r")
        plt.xlabel("time [s]")
        plt.ylabel("angle [deg]")
        # plt.plot(ex_dict[key]["t"], ex_dict[key]["force"]/10, color = "g", label = "force")
        # plt.legend()
    return

def plot_res_shank(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], color = "lightsalmon", label = "kma")
    plt.plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], color = "lightsteelblue", label = "shank")
    plot_sep_shank(data_shank["sub1"])
    # plt.plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "res shank")
    # plt.plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "res kma")
    # plt.plot(d1["t"][idx], d1["force"][idx]/10, color = "g", label = "force")
    # plt.plot(d1["t"][idx], d1["R_leg"][idx]*20, label = "R_leg")
    # plt.plot(d1["t"][idx], d1["L_leg"][idx]*20, label = "L_leg")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.title("Shank angle Subject 1")
    # plt.legend()
    return 

def plot_shank_force(d1):
    idx = range(10, len(d1["t"]) - 10)
    fig, axes = plt.subplots(2, sharex = True)
    axes[0].plot(d1["t"][idx], d1["force"][idx], color = "g", label = "force")
    axes[0].set_ylabel("Force [N]")
    axes[1].plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], color = "r", label = "kma")
    axes[1].plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], color = "b", label = "shank")
    axes[1].set_ylabel("angle [deg]")
    plt.xlabel("time [s]")
    fig.suptitle("Shank angle Subject 1")

plot_res_shank(data1)
# plot_shank_force(data1)

def plot_sep_thigh(ex_dict):
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
    for key in ex_dict:
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_thigh_angle"], color = "b")
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmau_angle"], color = "r")
        plt.xlabel("time [s]")
        plt.ylabel("angle [deg]")
        # plt.plot(ex_dict[key]["t"], ex_dict[key]["force"]/10, color = "g", label = "force")
        # plt.legend()
    return

# plot_sep_shank(data_shank["sub1"])
# plot_sep_shank(data_shank["sub3"])
# plot_sep_shank(data_shank["sub4"])
# plot_sep_shank(data_shank["sub5"])
# plot_sep_shank(data_shank["sub6"])

def plot_shank_tog(ex_dict):
    plt.figure()
    for key in ex_dict:
        plot_sep_shank(ex_dict[key])
    return 

# plot_shank_tog(data_shank)

def plot_thigh_tog(ex_dict):
    plt.figure()
    for key in ex_dict:
        plot_sep_thigh(ex_dict[key])
    return 

# plot_thigh_tog(data_thigh)

# def plot_sep_thigh(ex_dict):
#     """
#     plots each gait cycle separetely to form whole data
#     - to check that individual gaits were cut correctly 
    
#     Parameters
#     ----------
#     ex_dict : dictionary of dataframe, each containing individual gait cycle

#     Returns
#     -------
#     None.

#     """
#     plt.figure()
#     for key in ex_dict:
#         plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_thigh_angle"], label = key)
#         plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmau_angle"], label = key)
#         plt.legend()
#     return

# plot_sep_thigh(data_thigh)

def plot_zero(ex_dict):
    plt.figure()
    for key in ex_dict:
        plt.plot(ex_dict[key]["t"], ex_dict[key]["force"]/100, color = "g", label = "force")
        plt.plot(ex_dict[key]["t"], ex_dict[key]["shank_angle_grad"], color = "b")
        plt.plot(ex_dict[key]["t"], ex_dict[key]["kmal_angle_grad"], color = "r")
        # plt.plot(ex_dict[key]["t"], ex_dict[key]["filt_shank_angle"], color = "b")
        # plt.plot(ex_dict[key]["t"], ex_dict[key]["filt_kmal_angle"], color = "r")
        plt.title("mis a 0 sub" + str(key))
        # plt.legend()
    return
#%%
def plot_sub16(dict1, dict6):
    fig, (axes1, axes2) = plt.subplots(1, 2, constrained_layout = True, sharey=True)
    for key in dict1:
        axes1.plot(dict1[key]["t"], dict1[key]["no_mc_shank_angle"], color = "b", label = "shank")
        axes1.plot(dict1[key]["t"], dict1[key]["no_mc_kmal_angle"], color = "r", label = "kmal")
        # axes1.legend()
        axes1.set_xlabel("time [s]")
        axes1.set_ylabel("angles [deg]")
        axes1.set_title("Subject 1")
    for key in dict6:
        axes2.plot(dict6[key]["t"], dict6[key]["no_mc_shank_angle"], color = "b", label = "shank")
        axes2.plot(dict6[key]["t"], dict6[key]["no_mc_kmal_angle"], color = "r", label = "kmal")
        # axes2.legend()
        axes2.set_xlabel("time [s]")
        axes2.set_title("Subject 6")
    fig.suptitle("Example of shank gait cycles", fontsize = 16)
    return

plot_sub16(data_shank["sub1"], data_shank["sub6"])


#%%

def plot_subthigh16(dict1, dict6):
    fig, (axes1, axes2) = plt.subplots(1, 2, constrained_layout = True, sharey=True)
    for key in dict1:
        axes1.plot(dict1[key]["t"], dict1[key]["no_mc_thigh_angle"], color = "b", label = "thigh")
        axes1.plot(dict1[key]["t"], dict1[key]["no_mc_kmau_angle"], color = "r", label = "kmau")
        # axes1.plot(dict1[key]["t"], dict1[key]["force"]/10, color = "g", label = "force")
        # axes1.legend()
        axes1.set_xlabel("time [s]")
        axes1.set_ylabel("angles [deg]")
        axes1.set_title("Subject 1")
    for key in dict6:
        axes2.plot(dict6[key]["t"], dict6[key]["no_mc_thigh_angle"], color = "b", label = "thigh")
        axes2.plot(dict6[key]["t"], dict6[key]["no_mc_kmau_angle"], color = "r", label = "kmau")
        # axes2.plot(dict6[key]["t"], dict6[key]["force"]/10, color = "g", label = "force")
        # axes2.legend()
        axes2.set_xlabel("time [s]")
        axes2.set_title("Subject 3")
    fig.suptitle("Example of thigh gait cycles", fontsize = 16)
    return

plot_subthigh16(data_thigh["sub1"], data_thigh["sub3"])

#%%
def cut_cut_init_data(dict_in):
    cut_cut_dict = {}
    for key in dict_in:
        cut_cut_dict[key] = dict_in[key].loc[12:]
    return cut_cut_dict

def cut_init_data(data_in):
    cut_dict = {}
    for key in data_in:
        cut_dict[key] = cut_cut_init_data(data_in[key])
    return cut_dict

cut_shank_data = cut_init_data(data_shank)
#%%  

def butter_lowpass(cutoff, fs, order = 2):
    cutoff = 7 #cutoff frequency in Hz    
    nyq  = 0.5 * fs #determine Nyquist frequency    
    order = 2 #order of butterworth filter    
    normal_cutoff = cutoff /  nyq 
    b,a = scipy.signal.butter(order,normal_cutoff,btype='low',analog=False)
    return b,a 
    
def butter_lowpass_flter(data, cutoff,fs, order =2):
    b,a = butter_lowpass(cutoff, fs, order = order)
    y = scipy.signal.lfilter(b,a,data)
    return y 

def get_filt_shank(data_in):
    
    for key in data_in:
        fs = 100 #(len(data_in[key]['t'])-1) / np.amax(data_in[key]['t']) #Sampling frequency in Hz
        order = 2
        cutoff = 7
        
        filtered_shank = butter_lowpass_flter(data_in[key]["no_mc_shank_angle"], cutoff, fs)
        filtered_kmal = butter_lowpass_flter(data_in[key]["no_mc_kmal_angle"], cutoff, fs)
        
        data_in[key]["filt_shank_angle"] = filtered_shank
        data_in[key]["filt_kmal_angle"] = filtered_kmal
    return 

def get_deriv(data_in):
    for key in data_in:
                
        grad_shank = np.gradient(data_in[key]["filt_shank_angle"])
        grad_kmal = np.gradient(data_in[key]["filt_kmal_angle"])
        
        data_in[key]["shank_angle_grad"] = grad_shank
        data_in[key]["kmal_angle_grad"] = grad_kmal
        
    return 

get_filt_shank(data_shank["sub1"])
get_filt_shank(data_shank["sub3"])
get_filt_shank(data_shank["sub4"])
get_filt_shank(data_shank["sub5"])
get_filt_shank(data_shank["sub6"])

get_deriv(cut_shank_data["sub1"])
get_deriv(cut_shank_data["sub3"])
get_deriv(cut_shank_data["sub4"])
get_deriv(cut_shank_data["sub5"])
get_deriv(cut_shank_data["sub6"])


def c_shank(data_in):
    for key in data_in:
        plot_zero(data_in[key])
    return 

c_shank(cut_shank_data)
# c_shank(data_shank)
