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
        
#%%


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

#%% functions to cut 

# find max
def dyn_places_to_cut(d1):
    """
    finds maximum of shank angle -- corresponds to heel strike

    Parameters
    ----------
    d1 : dataframe containing dynamic trial

    Returns
    -------
    list_of_max : list, contains indices of max values, 0 and last value added 
    to help cut the data later

    """
    max2 = scipy.signal.argrelextrema(d1["mc_shank_angle"].values, np.greater_equal, order = 100)
    size_data = len(d1)
    list_of_max =  max2[0]
    list_of_max = np.append(list_of_max, size_data)
    list_of_max = np.append(0, list_of_max)
    return list_of_max


def dyn_cut_in_ind_gait(d1, list_of_max):
    """
    cuts dynamic trial into individual gait cycles 

    Parameters
    ----------
    d1 : dataframe containing dynamic trial
    list_of_max : list of indices of max shank values 

    Returns
    -------
    None. -- dictionnary containing dataframe of each ind gait cycle 

    """
    dict_of_df = {}
    for cidx in range(len(list_of_max)-1):
        dict_of_df[cidx] = d1.iloc[list_of_max[cidx]:list_of_max[cidx+1]]  
        
    for key in dict_of_df:
        dict_of_df[key] = dict_of_df[key].reset_index()
    
    return  dict_of_df


def dyn_cut_to_ind_gait_cycle(data_in):
    """
    separates dynamic dataframe into dict of individual gait cycle

    Parameters
    ----------
    data_in : dataframe : dynamic experiment 

    Returns
    -------
    dict_ind_gait : dict : dict of dataframe of separated gait cycle

    """
    l_max = dyn_places_to_cut(data_in)
    dict_ind_gait = dyn_cut_in_ind_gait(data_in, l_max)
    print(" dictionary of individual gait cycles created")    
    return dict_ind_gait

#%% test 

# tcut_filein = r"E:\ETHZ\mast_sem_IV\pdm\code02\T_INT.pkl"
# cut_file_name = "T_CUT.pkl"
# d_in = op_pickle(tcut_filein)

# dyn_remove_offset_bodypart(d_in, "shank")
# dyn_remove_offset_bodypart(d_in, "thigh")

# cut_dict = dyn_cut_to_ind_gait_cycle(d_in)

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

#%%
# plot_single_gait(dict_of_df, 40)
# plot_single_gait(dict_of_df, 50)

