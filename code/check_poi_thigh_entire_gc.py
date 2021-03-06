# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:00:00 2021

@author: ksdiv

points of interest THIGH

input has to be uncut data ! 
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
        

#%% cutting functions 


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



def cut_data(d1, delay_idx_r, delay_idx_l, steps_to_cut, nbr_sub):
    """
    cut data such that left with ~ heel strike to toe off for right leg 
    -- corresponds approx. to heel strike right leg to heel strike left leg 
    (s.t) seen from MyoSuit -- corresponds to where force is applied 

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

# p = cut_data(data1, 5, 2, 1)

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

def find_flat_kmau(d1):
    """
    returns FALSE if finds some errors in residual data (when residual = 0)
    for kmau 
    Parameters
    ----------
    d1 : panda dataframe : of individual gait cycle (or whole dyn data)

    Returns
    -------
    bool : TRUE if ok, FALSE if error

    """
    nbr_zeros = (d1.res_norm_kmau == 0).sum()
    return nbr_zeros < 2

def find_flat_thigh(d1):
    """
    returns FALSE if finds some errors in residual data (when residual = 0)
    for thigh
    Parameters
    ----------
    d1 : panda dataframe : of individual gait cycle (or whole dyn data)

    Returns
    -------
    bool : TRUE if ok, FALSE if error

    """
    nbr_zeros = (d1.res_norm_thigh == 0).sum()
    return nbr_zeros < 2

def remove_bad_kmau_val(dict_in, value):
    """
    removes (partial) gait cycles which have kmau residuals out of certain
    thresholds, and if kmau residuals = 0 

    Parameters
    ----------
    dict_in : dict of pd dataframe : dict containing ind. gait cycles
    value : upper bound to cut kmau residuals 

    Returns
    -------
    cut_dict : dict containing remaining gait cycles 

    """
    cut_dict = {}
    
    for key in dict_in: 
        max_res_kmau = dict_in[key]["res_norm_kmau"].max()
        min_res_kmau = dict_in[key]["res_norm_kmau"].min()
        
        if 0 <= max_res_kmau <= value:
            if 0 <= min_res_kmau <= value:
                if find_flat_kmau(dict_in[key]):
                    cut_dict[key] = dict_in[key]
                                        
    print("# of gait cycles, kmau removed : %f " %len(cut_dict))
    
    return cut_dict

# pkmau = remove_bad_kmau_val(p, 1.2)

def remove_bad_thigh_val(dict_in, value_max, value_min):
    """
    removes gait cycles with "bad" thigh residuals
    1. if residuals out of certain bounds 
    (for moment for thigh don't look at slope)
    
    Parameters
    ----------
    dict_in : dict pd dataframe : contains (partial) gait cycles HS -> TO
    value_max : float : max value that residual can have
    value_min : float : max value that lowest value of residual can have
    
    Returns
    -------
    out_dict : dict pd dataframe : contains "surviving" gait cycles 
    """
    
    out_dict = {}
    temp_t = []
    temp_kmau = []
    temp_thigh = []
    temp_res_k = []
    temp_res_t = []
    
    for key in dict_in: 
        max_res_thigh = dict_in[key]["res_norm_thigh"].max()
        min_res_thigh = dict_in[key]["res_norm_thigh"].min()
        
        if 0 <= max_res_thigh <= value_max:
            if 0 <= min_res_thigh <= value_min:
                if find_flat_thigh(dict_in[key]):
                    out_dict[key] = dict_in[key]
                    temp_t.append(dict_in[key]["t"].to_list())
                    temp_kmau.append(dict_in[key]["no_mc_kmau_angle"].to_list())
                    temp_thigh.append(dict_in[key]["no_mc_thigh_angle"].to_list())
                    temp_res_k.append(dict_in[key]["res_norm_kmau"].to_list())
                    temp_res_t.append(dict_in[key]["res_norm_thigh"].to_list())
                    # out_list.append([dict_in[key]["t"],dict_in[key]["no_mc_kmau_angle"],
                                    # dict_in[key]["no_mc_thigh_angle"], dict_in[key]["res_norm_kmau"],
                                    # dict_in[key]["res_norm_thigh"]])
    flat_temp_t = [item for sublist in temp_t for item in sublist]  
    flat_temp_kmau = [item for sublist in temp_kmau for item in sublist] 
    flat_temp_thigh = [item for sublist in temp_thigh for item in sublist]  
    flat_temp_res_k = [item for sublist in temp_res_k for item in sublist] 
    flat_temp_res_t = [item for sublist in temp_res_t for item in sublist]       
    temp_dict = {"t": flat_temp_t, "no_mc_kmau_angle" : flat_temp_kmau, "no_mc_thigh_angle" : flat_temp_thigh,
                 "res_norm_kmau" : flat_temp_res_k, "res_norm_thigh" : flat_temp_res_t}         
    out_list_df = pd.DataFrame(data = temp_dict)
    print("# of gait cycles, thigh removed : %f " %len(out_dict))
                    
    return out_dict, out_list_df


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
    
#%% plots

def plot_res_thigh(d1, sub_nbr):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmau_angle"][idx], label = "kma")
    plt.plot(d1["t"][idx], d1["no_mc_thigh_angle"][idx], label = "body part")
    plt.plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "res thigh")
    plt.plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "res kmau")
    # plt.plot(d1["t"][idx], d1["force"][idx]/10, label = "force/10")
    # plt.plot(d1["t"][idx], d1["R_leg"][idx]*20, label = "R_leg")
    # plt.plot(d1["t"][idx], d1["L_leg"][idx]*20, label = "L_leg")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.title("Thigh S" + str(sub_nbr))
    plt.legend()
    return 



def plot_diff_force_mocap_data_thigh(d1):
    diff = d1["no_mc_thigh_angle"]-d1["no_mc_kmau_angle"]
    
    idx = range(10, len(d1["t"]) - 10)
    fig, axes = plt.subplots(2, sharex= True)
    axes[0].plot(d1["t"][idx],diff[idx], label = "thigh - kmau angle ")
    axes[0].set_ylabel("angle difference [degree]")
    axes[0].set_title("Mocap estimated (thigh - KMA) angles")
    axes[1].plot(d1["t"][idx], d1["force"][idx], label = "force")
    axes[1].set_ylabel("force [N]")
    axes[1].set_title("Applied force")
    fig.suptitle("Applied force vs. difference in mocap estimated angles thigh")
    plt.xlabel("time [s]")
    # plt.legend()
    return

# plot_diff_force_mocap_data_thigh(data1)


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
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_thigh_angle"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmau_angle"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["res_norm_thigh"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["res_norm_kmau"], label = key)
        plt.legend()
    return

def plot_whole(df):
    plt.figure()
    plt.plot(df["t"], df["no_mc_thigh_angle"], label = "thigh angle")
    plt.plot(df["t"], df["no_mc_kmau_angle"], label = "kmau angle")
    plt.plot(df["t"], df["res_norm_thigh"], label = "res thigh")
    plt.plot(df["t"], df["res_norm_kmau"], label = "res kmau")
    plt.title("gait cycles thigh og")
    plt.legend()
    return

def plot_quick(df):
    plt.figure()
    plt.plot(df["t"], df["no_mc_thigh_angle"],".", label = "thigh angle")
    plt.plot(df["t"], df["no_mc_kmau_angle"],".", label = "kmau angle")
    plt.plot(df["t"], df["res_norm_thigh"],".", label = "res thigh")
    plt.plot(df["t"], df["res_norm_kmau"],".", label = "res kmau")
    plt.title("individual gait cycles thigh after cut")
    plt.legend()
    return

def plot_res_kmau(d1,d3,d4,d5,d6):
    idx = range(10, len(d1["t"]) - 10)
    
    fig, axes = plt.subplots(5, sharex = True)
    axes[0].plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "S1")
    axes[0].legend()
    axes[0].set_ylabel("residuals [-]")
    axes[1].plot(d3["t"][idx], d3["res_norm_kmau"][idx], label = "S3")
    axes[1].legend()
    axes[1].set_ylabel("residuals [-]")
    axes[2].plot(d4["t"][idx], d4["res_norm_kmau"][idx], label = "S4")
    axes[2].legend()
    axes[2].set_ylabel("residuals [-]")
    axes[3].plot(d5["t"][idx], d5["res_norm_kmau"][idx], label = "S5")
    axes[3].legend()
    axes[3].set_ylabel("residuals [-]")
    axes[4].plot(d6["t"][idx], d6["res_norm_kmau"][idx], label = "S6")
    axes[4].legend()
    axes[4].set_ylabel("residuals [-]")
    fig.suptitle("residuals KMAU, all subjects")
    plt.xlabel("time [s]")
    return

# plot_res_kmau(data1,data3,data4,data5,data6)

def plot_resi_thigh(d1,d3,d4,d5,d6):
    idx = range(10, len(d1["t"]) - 10)
    
    fig, axes = plt.subplots(5, sharex = True)
    axes[0].plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "S1")
    axes[0].legend()
    axes[0].set_ylabel("residuals [-]")
    axes[1].plot(d3["t"][idx], d3["res_norm_thigh"][idx], label = "S3")
    axes[1].legend()
    axes[1].set_ylabel("residuals [-]")
    axes[2].plot(d4["t"][idx], d4["res_norm_thigh"][idx], label = "S4")
    axes[2].legend()
    axes[2].set_ylabel("residuals [-]")
    axes[3].plot(d5["t"][idx], d5["res_norm_thigh"][idx], label = "S5")
    axes[3].legend()
    axes[3].set_ylabel("residuals [-]")
    axes[4].plot(d6["t"][idx], d6["res_norm_thigh"][idx], label = "S6")
    axes[4].legend()
    axes[4].set_ylabel("residuals [-]")
    fig.suptitle("residuals thigh, all subjects")
    plt.xlabel("time [s]")
    return

# plot_resi_thigh(data1,data3,data4,data5,data6)

#%% statistics

def get_median_res_kmau(data_in):
    return statistics.median(data_in["res_norm_kmau"])

def get_mad_res_kmau(data_in):
    #more robust to outliers
    return stats.median_abs_deviation(data_in["res_norm_kmau"])

# mad1= get_mad_res_kmau(data1)
# mad3= get_mad_res_kmau(data3)
# mad4= get_mad_res_kmau(data4)
# mad5= get_mad_res_kmau(data5)
# mad6= get_mad_res_kmau(data6)

# med1= get_median_res_kmau(data1)
# med3= get_median_res_kmau(data3)
# med4= get_median_res_kmau(data4)
# med5= get_median_res_kmau(data5)
# med6= get_median_res_kmau(data6)

#%%  main 

def keep_good_gc_thigh(data_in):
    
    delay_to_cut = 30 #10 
    delay_to_cut_l = 15
    nbr_steps_to_cut = 5
    min_val_thigh = 6
            
    # plot_whole(data_in)
    # gait cycle coup?? dans cette fonction
    dict_data_to_cut = cut_data(data_in, delay_to_cut, delay_to_cut_l, nbr_steps_to_cut, 25)
    if not bool(dict_data_to_cut) :
        # can be due to the fact that subject has 2 feet on ground at same time
        print("inital dict empty, trying again ")
        dict_data_to_cut = cut_data(data_in, delay_to_cut, delay_to_cut_l, nbr_steps_to_cut, 6)
    
    getMaxDict(dict_data_to_cut)
    
    medianmaxkmau = getMedianMaxDict(dict_data_to_cut, "res_norm_kmau")
    print("median of max kmau : %f " %medianmaxkmau)
    medianmaxkmau = m.ceil(medianmaxkmau) + 1
    print("cutting upper bound kmau : %f" %medianmaxkmau)
    
    reset_idx(dict_data_to_cut)
    # dict_cut, df_kmau = remove_bad_kmau_val(dict_data_to_cut, bound_res_kmau)
    dict_cut = remove_bad_kmau_val(dict_data_to_cut, medianmaxkmau)
    
    medianmaxthigh = getMedianMaxDict(dict_cut, "res_norm_thigh")
    print("median of max thigh : %f " %medianmaxthigh)
    medianmaxthigh = m.ceil(medianmaxthigh) + 1
    print("cutting upper bound thigh : %f" %medianmaxthigh)
    
    # final_cut_data, final_df = remove_bad_thigh_val(dict_cut, max_val_thigh, min_val_thigh)
    final_cut_data, final_df = remove_bad_thigh_val(dict_cut, medianmaxthigh, min_val_thigh)
        
    # plot_sep_thigh(final_cut_data)
    plot_quick(final_df)
    # plot_sep_thigh(dict_data_to_cut)
    
    return final_cut_data, final_df
    # return dict_cut







