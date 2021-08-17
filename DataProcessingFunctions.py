# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:32:28 2021

@author: ksdiv
removes offset 
cut dynamic trial into individual gait cycles 
"""

import pickle
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import pandas as pd
import statistics

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
#%% Remove offset leg related angles

def dyn_remove_offset_bodypart(data_in, body_part):
    """
    will get rid of the offset between the extraced angles from mocap data for 
    thigh/kmau or shank/kmal - so that can compare angles better
    for dynamic data works well, for static data better to remove offset for
    each position separately 

    Parameters
    ----------
    data_in : pd dataframe : dynamic experiment for one subject 
    body_part : "thigh" / "shank"

    Returns
    -------
    None. (changes data frame directly)

    """
    
    kma = "mc_kmal_angle"
    part = "mc_shank_angle"
    if body_part == "thigh":
        kma = "mc_kmau_angle"
        part = "mc_thigh_angle"
        
    mean1 = statistics.mean(data_in[kma][3:400])
    mean2 = statistics.mean(data_in[part][3:400])
    
    data_in["no_"+kma] = data_in[kma] - mean1
    data_in["no_"+part] = data_in[part] - mean2
    
    data_in_copy = data_in.copy()
    
    data_in_copy[kma] = data_in_copy[kma] - mean1
    data_in_copy[part] = data_in_copy[part] - mean2
    
    # return data_in_copy
    return

def se_fl1(data_in):
    data_in["no_mc_thigh_angle"] = - data_in["no_mc_thigh_angle"]
    return

#%% functions to cut trial

def find_mode_change(d_in):
    """
    function to separate the different sections of one trial (1/2). Does this by 
    finding the changes in Mode from MyoSuit data.

    Parameters
    ----------
    d_in : dataframe : whole trial 

    Returns
    -------
    idx_mode_change : list : contains indices of changes in Mode
    """
    d_in["Mode"] = d_in["Mode"].round()
    mode_change = d_in["Mode"].shift() != d_in["Mode"]
    idx_mode_change = mode_change.index[mode_change == True].tolist()
    print("nbr of mode changes = %i" %len(idx_mode_change))
    
    # plt.figure()
    # plt.plot(d_in["Mode"])
    # plt.plot(idx_mode_change, d_in["Mode"][idx_mode_change],"o")
    
    return idx_mode_change


def separate_data(d_in, idx_mode_change):
    """
    function to separate the different sections of one trial (2/2). Separates
    the input dataframe in relation to mode value and the structure of the trial.

    Parameters
    ----------
    d_in : dataframe : whole trial 
    idx_mode_change : list : output of find_mode_change, indices of mode change 

    Returns
    -------
    dict_data : dict : contains the separated trials

    """
    # normal
    transp = d_in[idx_mode_change[0]: idx_mode_change[1]]
    iso1 = d_in[idx_mode_change[1]: idx_mode_change[5]]
    concentric1 = d_in[idx_mode_change[5]: idx_mode_change[6]]
    lag_val = idx_mode_change[8] - idx_mode_change[7]
    iso2 = d_in[idx_mode_change[6]: (idx_mode_change[9] + lag_val)]
    concentric2 = d_in[idx_mode_change[9]:]
    dict_data =  {"Transparent" : transp, "Isometric1" : iso1, "Concentric1" : concentric1,
                  "Isometric2": iso2, "Concentric2" : concentric2}
    
    # special SE_FL5_2
    # transp = d_in[idx_mode_change[0]: idx_mode_change[1]]
    # iso1 = d_in[idx_mode_change[1]: idx_mode_change[5]]
    # concentric1 = d_in[idx_mode_change[5]: idx_mode_change[6]]
    # iso2 = d_in[idx_mode_change[1]: (idx_mode_change[5])]
    # concentric2 = d_in[idx_mode_change[5]:]
    # iso3 = d_in[idx_mode_change[12]:idx_mode_change[17]]
    # concentric3 = d_in[idx_mode_change[17]:]
    # # transp2 = d_in[idx_mode_change[18]:]
    # dict_data =  {
    #               "Isometric2": iso2, "Concentric2" : concentric2}
    
    # SE_FL5_1
    # transp = d_in[idx_mode_change[0]: idx_mode_change[1]]
    # iso1 = d_in[idx_mode_change[1]: idx_mode_change[5]]
    # concentric1 = d_in[idx_mode_change[5]: ]
    # dict_data =  {"Transparent" : transp, "Isometric1" : iso1, "Concentric1" : concentric1}
    
    return dict_data


def check_mode(dict_data):    
    """
    prints mean value of mode for each separated trial. 
    and time and FL values

    Parameters
    ----------
    dict_data : dict : contains separated trials

    Returns
    -------
    None.

    """
    print("Checking values of separated trials : ")
    for key in dict_data:
        mode_check = statistics.mean(dict_data[key]["Mode"])
        fl_check = statistics.mean(dict_data[key]["ForceLevel"])
        t_start = dict_data[key]["t"].iloc[0]
        t_end = dict_data[key]["t"].iloc[-1]
        tot_t = t_end - t_start
        print( key)
        print("length of trial : %f" % tot_t)
        print("mean Force Level : %f" % fl_check)
        print("mean mode : %f" % mode_check)
    return



# this one that calls all others 
def separate_trial(data_in): 
    """
    Separates the trial at a certain force level into the transparent, isometric
    and concentric parts. 
    Prints mode mean values to check. 

    Parameters
    ----------
    data_in : dataframe : whole trial, offset removed 

    Returns
    -------
    sep_data : dict : contains dataframe of separated trial 

    """
    mode_change_idx = find_mode_change(data_in)
    sep_data = separate_data(data_in, mode_change_idx)
    check_mode(sep_data)
    
    return sep_data


def save_all_bits_sep(dict_data, dirName):
    for key in dict_data:
        tmp_name = dirName + "_" + key
        save_obj(dict_data[key], tmp_name)
    return

#%% plots

def plot_sep(ex_dict):
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
        plt.plot(ex_dict[key]["t"], ex_dict[key]["mc_shank_angle"], label = key)
        plt.legend()
    return
    

def plot_single_gait(ex_dict, gait_nbr):
    plt.figure()
    plt.plot(ex_dict[gait_nbr]["t"], ex_dict[gait_nbr]["mc_shank_angle"], label = "shank")
    plt.plot(ex_dict[gait_nbr]["t"], ex_dict[gait_nbr]["mc_kmal_angle"], label = "kma l")
    plt.plot(ex_dict[gait_nbr]["t"], ex_dict[gait_nbr]["mc_thigh_angle"], label = "thigh")
    plt.plot(ex_dict[gait_nbr]["t"], ex_dict[gait_nbr]["mc_kmau_angle"], label = "kma u")
    plt.legend()
    plt.title("one gait cycle")
    return

def plot_res_shank(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], label = "kmal")
    plt.plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], label = "shank")
    plt.plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "res shank")
    plt.plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "res kma")
    # plt.plot(d1["t"][idx], d1["mc_kmal_angle"][idx], label = "kma")
    # plt.plot(d1["t"][idx], d1["mc_shank_angle"][idx], label = "body part")
    # plt.plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "res shank")
    # plt.plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "res kma")
    plt.title("shank angles mocap")
    plt.legend()
    return 

def plot_res_thigh(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmau_angle"][idx], label = "kmau")
    plt.plot(d1["t"][idx], d1["no_mc_thigh_angle"][idx], label = "thigh")
    plt.plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "res thigh")
    plt.plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "res kmau")
    
    # plt.plot(d1["t"][idx], d1["mc_kmau_angle"][idx], label = "kma")
    # plt.plot(d1["t"][idx], d1["mc_thigh_angle"][idx], label = "body part")
    # plt.plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "res thigh")
    # plt.plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "res kmau")
    plt.title("thigh angles mocap")
    plt.legend()
    return 

#%%


