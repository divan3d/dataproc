# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:15:45 2021

@author: ksdiv
get whole gait cycle 
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
import pandas as pd

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
def indices_to_cut_R(d1):
    """
    finds indices to cut data such that left with ~ heel strike to toe off for 
    right leg -- corresponds approx. to heel strike right leg to heel strike left leg 

    Parameters
    ----------
    d1 : pd dataframe : dynamic data of 1 subject

    Returns
    -------
    peak_R_leg : array : contains indices of Right leg heel strike 

    """
    _, properties = scipy.signal.find_peaks(d1["R_leg"].values, height=1, plateau_size = 5)
    peak_R_leg = properties["left_edges"]
    return peak_R_leg

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


def cut_data(d1, delay_idx_r, delay_idx_l, steps_to_cut, nbr_sub):
    """
    cut data such that get gait cycle for right leg (HS -> HS)

    Parameters
    ----------
    d1 : pd dataframe : dynamic data of 1 subject
    delay_idx : int : how many indices before for right leg, after for left leg
                        to cut 
    steps_to_cut : int : first and last few steps of dynamic data are corrupted
                        so cut them out directly
     
    Returns
    -------
    dict_cut_data : dict of dataframe : each dict has dataframe of one gait cycle
                        from ~ heel strike to ~ toe off

    """
    dict_cut_data = {}
    
    idx_R_leg = indices_to_cut_R(d1)
        
    # idx_R_leg = dyn_places_to_cut(d1)
    
    
    # for cidx in range(len(idx_R_leg) - 2*steps_to_cut):
    #     dict_cut_data[cidx] = d1.iloc[idx_R_leg[cidx + steps_to_cut]: idx_R_leg[cidx + 1 + steps_to_cut]] 
    
    # with offset 
    offset = 5 # for 50 ms 
    idx_r1 = idx_R_leg - offset
    idx_r2 = idx_R_leg + offset 
    
    for cidx in range(len(idx_R_leg) - 2*steps_to_cut):
        dict_cut_data[cidx] = d1.iloc[idx_r1[cidx + steps_to_cut]: idx_r2[cidx + 1 + steps_to_cut]] 
    
    print("# of gait cycles, initial cut : %f " %len(dict_cut_data))
    
    return dict_cut_data

#%% cleaning functions 

def reset_idx(dict_of_df):
    """
    for each dataframe in dict, starts the indexing from 0 to #, keeps old
    index in separate column 

    Parameters
    ----------
    dict_of_df : dict of pd dataframe : contains (partial) gait cycles 

    Returns
    -------
    None.

    """
    for key in dict_of_df:
        dict_of_df[key] = dict_of_df[key].reset_index()
    return

def find_flat_kmal(d1):
    """
    returns FALSE if finds some errors in residual data (when residual = 0)
    for kmal 
    Parameters
    ----------
    d1 : panda dataframe : of individual gait cycle (or whole dyn data)

    Returns
    -------
    bool : TRUE if ok, FALSE if error

    """
    nbr_zeros = (d1.res_norm_kmal == 0).sum()
    return nbr_zeros < 2

def find_flat_shank(d1):
    """
    returns FALSE if finds some errors in residual data (when residual = 0)
    for shank
    Parameters
    ----------
    d1 : panda dataframe : of individual gait cycle (or whole dyn data)

    Returns
    -------
    bool : TRUE if ok, FALSE if error

    """
    nbr_zeros = (d1.res_norm_shank == 0).sum()
    return nbr_zeros < 2

def remove_bad_kmal_val(dict_in, value):
    """
    removes (partial) gait cycles which have kmal residuals out of certain
    thresholds, and if kmal residuals = 0 

    Parameters
    ----------
    dict_in : dict of pd dataframe : dict containing ind. gait cycles
    value : upper bound to cut kmal residuals 

    Returns
    -------
    cut_dict : dict containing remaining gait cycles 

    """
    cut_dict = {}
    for key in dict_in: 
        max_res_kmal = dict_in[key]["res_norm_kmal"].max()
        min_res_kmal = dict_in[key]["res_norm_kmal"].min()
        
        if 0 <= max_res_kmal <= value:
            if 0 <= min_res_kmal <= value:
                if find_flat_kmal(dict_in[key]):
                    cut_dict[key] = dict_in[key]
                    
    print("# of gait cycles, kmal removed : %f " %len(cut_dict))
    
    return cut_dict

def smooth_cut(dict_in):
    """
    uses hann window to smooth shank residuals 

    Parameters
    ----------
    dict_in : dict of pd dataframe : contains individual (partial) gait cycles

    Returns
    -------
    val_smoo : dict of pd dataframe with smoothed residual norm shank

    """
    win = scipy.signal.windows.hann(50)
    val_smoo = {}
    for key in dict_in:
        filtered = scipy.signal.convolve(dict_in[key]["res_norm_shank"], win, mode = "same")/ sum(win)
        val_smoo[key] = {}
        val_smoo[key]["res_norm_shank_filtered"] = filtered 
    return val_smoo
 
def grad_cut(dict_in):
    """
    gets gradient 

    Parameters
    ----------
    dict_in : dict pd dataframe : containing smoothed values

    Returns
    -------
    val_grad : dict pd dataframe : contains gradient of input

    """
    val_grad = {}
    for key in dict_in:
        grad = np.gradient(dict_in[key]["res_norm_shank_filtered"])
        val_grad[key] = {}
        val_grad[key]["gradient"] = grad
    return val_grad   

def remove_bad_shank_val(dict_in, value_max, value_min, slope_min):
    """
    removes gait cycles with "bad" shank residuals
    1. if residuals out of certain bounds
    2. if residual doesn't follow certain shape - want the residual near toe off
        to be flat -- smooths and derives res_norm_shank to check this 

    Parameters
    ----------
    dict_in : dict pd dataframe : contains (partial) gait cycles HS -> TO
    value_max : float : max value that residual can have
    value_min : float : max value that lowest value of residual can have
    slope_min : float : min bound of derivative of res_norm_shank - to impose 
                        flat curve near toe off (TO)
    
    Returns
    -------
    out_dict : dict pd dataframe : contains "surviving" gait cycles 
    """
    cut_dict = {}
    temp_t = []
    temp_kmal = []
    temp_shank = []
    temp_res_k = []
    temp_res_s = []
    # out_dict = {}
    for key in dict_in: 
        max_res_shank = dict_in[key]["res_norm_shank"].max()
        min_res_shank = dict_in[key]["res_norm_shank"].min()
        
        if 0 <= max_res_shank <= value_max:
            if 0 <= min_res_shank <= value_min:
                if find_flat_shank(dict_in[key]):
                    cut_dict[key] = dict_in[key]
                    temp_t.append(dict_in[key]["t"].to_list())
                    temp_kmal.append(dict_in[key]["no_mc_kmal_angle"].to_list())
                    temp_shank.append(dict_in[key]["no_mc_shank_angle"].to_list())
                    temp_res_k.append(dict_in[key]["res_norm_kmal"].to_list())
                    temp_res_s.append(dict_in[key]["res_norm_shank"].to_list())
                    
    flat_temp_t = [item for sublist in temp_t for item in sublist]  
    flat_temp_kmal = [item for sublist in temp_kmal for item in sublist] 
    flat_temp_shank = [item for sublist in temp_shank for item in sublist]  
    flat_temp_res_k = [item for sublist in temp_res_k for item in sublist] 
    flat_temp_res_s = [item for sublist in temp_res_s for item in sublist]       
    temp_dict = {"t": flat_temp_t, "no_mc_kmal_angle" : flat_temp_kmal, "no_mc_shank_angle" : flat_temp_shank,
                 "res_norm_kmal" : flat_temp_res_k, "res_norm_shank" : flat_temp_res_s}         
    out_list_df = pd.DataFrame(data = temp_dict)
                
    print("# of gait cycles, shank removed : %f " %len(cut_dict))
    
    # # slope
    # smoothed_val = smooth_cut(cut_dict)
    # grad_val = grad_cut(smoothed_val)
    
    # for key in grad_val:
    #     if slope_min < grad_val[key]["gradient"][-1]:
    #         out_dict[key] = cut_dict[key]
    
    return cut_dict, out_list_df

def getMaxDict(dict_in):
    # to get longuest GC once its been separated in cut data 
    print("longuest GC: (length, key)")
    print(max((len(v), k) for k,v in dict_in.items()))
    return


def getMedianMaxDict(dict_in, res_name):
    tp_list = []
    for key in dict_in: 
        tp_list.append(dict_in[key][res_name].max())
        
    medianmax = statistics.median(tp_list)
    return medianmax

def plot_quick(df):
    plt.figure()
    plt.plot(df["t"], df["no_mc_shank_angle"],".", label = "shank angle")
    plt.plot(df["t"], df["no_mc_kmal_angle"],".", label = "kmal angle")
    plt.plot(df["t"], df["res_norm_shank"],".", label = "res shank")
    plt.plot(df["t"], df["res_norm_kmal"],".", label = "res kmal")
    plt.title("individual gait cycles shank after cut")
    plt.legend()
    return

#%%
def keep_good_gc_shank(data_in):
    delay_to_cut_r = 15
    delay_to_cut_l = 30
    nbr_steps_to_cut = 5    
    # bound_res_kmal = 3
    # max_val_shank = 7 #5 
    min_val_shank = 4
    min_slope = -0.1
    
    # plot_whole(data_in)
    
    dict_data_to_cut = cut_data(data_in, delay_to_cut_r, delay_to_cut_l, nbr_steps_to_cut, 25)
    
    if not bool(dict_data_to_cut) :
       # can be due to the fact that subject has 2 feet on ground at same time
       print("inital dict empty, trying again ")
       dict_data_to_cut = cut_data(data_in, delay_to_cut_r, delay_to_cut_l, nbr_steps_to_cut, 6)
    
    getMaxDict(dict_data_to_cut)
    
    medianmaxkmal = getMedianMaxDict(dict_data_to_cut, "res_norm_kmal")
    print("median of max kmal : %f " %medianmaxkmal)
    medianmaxkmal = m.ceil(medianmaxkmal) + 1
    print("cutting upper bound kmal : %f" %medianmaxkmal)
    
    reset_idx(dict_data_to_cut)
    dict_cut = remove_bad_kmal_val(dict_data_to_cut, medianmaxkmal)
    # plot_sep(dict_cut)
    
    medianmaxshank = getMedianMaxDict(dict_cut, "res_norm_shank")
    print("median of max shank : %f " %medianmaxshank)
    medianmaxshank = m.ceil(medianmaxshank) + 1
    print("cutting upper bound shank : %f" %medianmaxshank)
    
    final_cut_data, final_df = remove_bad_shank_val(dict_cut, medianmaxshank, min_val_shank, min_slope)

    # plot_sep(final_cut_data)
    plot_quick(final_df)
    
    return final_cut_data, final_df


