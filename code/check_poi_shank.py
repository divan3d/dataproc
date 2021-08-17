# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:51:27 2021

@author: ksdiv
look at point of interest (poi)
SHANK
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
        
#%%
file1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_FL3_TEST\SA_FL3_TEST_Concentric2"
data1 = op_pickle(file1)
file3 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_FL3_TEST\SA_FL3_TEST_Concentric1"
data3 = op_pickle(file3)
file4 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_FL3_TEST\SA_FL3_TEST_Transparent"
data4 = op_pickle(file4)
# file5 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S05_Mocap_angles_removed_offset_res_MS.pkl"
# data5 = op_pickle(file5)
# file6 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S06_Mocap_angles_removed_offset_res_MS.pkl"
# data6 = op_pickle(file6)

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

def indices_to_cut_L(d1):
    """
    finds indices to cut data such that left with ~ heel strike to toe off for 
    right leg -- corresponds approx. to heel strike right leg to heel strike left leg 

    Parameters
    ----------
    d1 : pd dataframe : dynamic data of 1 subject

    Returns
    -------
    peak_L_leg : array : contains indices of Left leg heel strike 

    """
    _, properties = scipy.signal.find_peaks(d1["L_leg"].values, height=1, plateau_size = 5)
    peak_L_leg = properties["left_edges"]
    return peak_L_leg

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
    idx_L_leg = indices_to_cut_L(d1)
    
    # #S6 -- pose les deux pieds en meme temps 
    if nbr_sub == 6:
        idx_R_leg = idx_R_leg[1:]
        idx_L_leg = idx_L_leg[2:]
    
    idx_R_leg = idx_R_leg - delay_idx_r
    idx_L_leg = idx_L_leg + delay_idx_l
    
    if idx_L_leg[0] < idx_R_leg[0]:
        idx_L_leg = idx_L_leg[1:]
        idx_R_leg = idx_R_leg[:-1]
    for cidx in range(len(idx_R_leg) - 2*steps_to_cut):
        dict_cut_data[cidx] = d1.iloc[idx_R_leg[cidx + steps_to_cut]: idx_L_leg[cidx + steps_to_cut]]
    
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
    # out_dict = {}
    for key in dict_in: 
        max_res_shank = dict_in[key]["res_norm_shank"].max()
        min_res_shank = dict_in[key]["res_norm_shank"].min()
        
        if 0 <= max_res_shank <= value_max:
            if 0 <= min_res_shank <= value_min:
                if find_flat_shank(dict_in[key]):
                    cut_dict[key] = dict_in[key]
                
    # # slope
    # smoothed_val = smooth_cut(cut_dict)
    # grad_val = grad_cut(smoothed_val)
    
    # for key in grad_val:
    #     if slope_min < grad_val[key]["gradient"][-1]:
    #         out_dict[key] = cut_dict[key]
    
    return cut_dict


#%% plot functions

def plot_res_shank(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], label = "kma")
    plt.plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], label = "body part")
    # plt.plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "res shank")
    # plt.plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "res kma")
    # plt.plot(d1["t"][idx], d1["force"][idx]/10, label = "force")
    # plt.plot(d1["t"][idx], d1["R_leg"][idx]*20, label = "R_leg")
    # plt.plot(d1["t"][idx], d1["L_leg"][idx]*20, label = "L_leg")
    plt.title("shank")
    plt.legend()
    return 


# plot_res_shank(data1)
# plot_res_shank(data3)
# plot_res_shank(data4)
# plot_res_shank(data5)
# plot_res_shank(data6)

def plot_res_kmal(d1,d3,d4,d5,d6):
    idx = range(10, len(d1["t"]) - 10)
    
    fig, axes = plt.subplots(5, sharex = True)
    axes[0].plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "S1")
    axes[0].legend()
    axes[0].set_ylabel("residuals [-]")
    axes[1].plot(d3["t"][idx], d3["res_norm_kmal"][idx], label = "S3")
    axes[1].legend()
    axes[1].set_ylabel("residuals [-]")
    axes[2].plot(d4["t"][idx], d4["res_norm_kmal"][idx], label = "S4")
    axes[2].legend()
    axes[2].set_ylabel("residuals [-]")
    axes[3].plot(d5["t"][idx], d5["res_norm_kmal"][idx], label = "S5")
    axes[3].legend()
    axes[3].set_ylabel("residuals [-]")
    axes[4].plot(d6["t"][idx], d6["res_norm_kmal"][idx], label = "S6")
    axes[4].legend()
    axes[4].set_ylabel("residuals [-]")
    fig.suptitle("residuals KMAL, all subjects")
    plt.xlabel("time [s]")
    return

# plot_res_kmal(data1,data3,data4,data5,data6)

def plot_resi_shank(d1,d3,d4,d5,d6):
    idx = range(10, len(d1["t"]) - 10)
    
    fig, axes = plt.subplots(5, sharex = True)
    axes[0].plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "S1")
    axes[0].legend()
    axes[0].set_ylabel("residuals [-]")
    axes[1].plot(d3["t"][idx], d3["res_norm_shank"][idx], label = "S3")
    axes[1].legend()
    axes[1].set_ylabel("residuals [-]")
    axes[2].plot(d4["t"][idx], d4["res_norm_shank"][idx], label = "S4")
    axes[2].legend()
    axes[2].set_ylabel("residuals [-]")
    axes[3].plot(d5["t"][idx], d5["res_norm_shank"][idx], label = "S5")
    axes[3].legend()
    axes[3].set_ylabel("residuals [-]")
    axes[4].plot(d6["t"][idx], d6["res_norm_shank"][idx], label = "S6")
    axes[4].legend()
    axes[4].set_ylabel("residuals [-]")
    fig.suptitle("residuals shank, all subjects")
    plt.xlabel("time [s]")
    return

# plot_resi_shank(data1,data3,data4,data5,data6)

def plot_diff_force_mocap_data_shank(d1):
    """
    plots diff btwn kmal and shank with relation to force 

    Parameters
    ----------
    d1 : dataframe - dynamic data of 1 subject

    Returns
    -------
    None.

    """
    diff = d1["no_mc_kmal_angle"]-d1["no_mc_shank_angle"]
    
    idx = range(10, len(d1["t"]) - 10)
    plt.figure()
    plt.plot(d1["t"][idx],diff[idx], label = "kmal - shank angle ")
    plt.plot(d1["t"][idx], d1["force"][idx]/10, label = "force")
    plt.title("force vs diff mocap angles shank")
    plt.legend()
    return


# plot_diff_force_mocap_data_shank(data1)

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
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_shank_angle"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmal_angle"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["res_norm_shank"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["res_norm_kmal"], label = key)
        plt.legend()
    return


#%% brouillon
# a = indices_to_cut_R(data1)
# u = np.array(a[:])

# b = indices_to_cut_L(data1)

# a0 = np.zeros([len(a)])
# b0 = np.ones([len(b)])

# aa = np.column_stack((a,a0))
# bb = np.column_stack((b,b0))

# c = np.concatenate((aa,bb))
# cs = c[c[:,0].argsort()]
# cs = np.column_stack((cs,np.zeros(len(cs))))

# for tidx in range(len(cs)-1):
#     cs[tidx,2] = cs[tidx,1] + cs[tidx + 1, 1]
    
# u = np.argwhere(cs[:,2]>1)
# u0 = m.ceil((int(u[0]) + 2)/2) 
# ac = a[u0:]
# bc = b[u0:]

# plt.figure()
# idx = range(10, len(data1["t"]) - 10)
# plt.plot(data1["t"][idx], data1["R_leg"][idx], label = "R_leg")
# plt.plot(data1["t"][idx], data1["L_leg"][idx], label = "L_leg")
# plt.plot(data1["t"][ac], data1["R_leg"][ac], "ro")
# plt.plot(data1["t"][bc], data1["L_leg"][bc], "go")

# def get_fft_shank(d1):
#     N = len(d1["t"])
#     # shank_fft = fft(d1["no_mc_shank_angle"])
#     y = d1["no_mc_shank_angle"].to_numpy()
#     res_shank_fft = fft(y)
#     xfft = fftfreq(N)
    
#     fig, axes = plt.subplots(2)
#     axes[0].plot(d1["t"], d1["no_mc_shank_angle"])
#     axes[1].plot(abs(xfft), res_shank_fft)
#     return res_shank_fft

#%% statistics 

def get_mean_res_kmal(data_in):
    return statistics.mean(data_in["res_norm_kmal"])

def get_var_res_kmal(data_in):
    return statistics.variance(data_in["res_norm_kmal"])

def get_median_res_kmal(data_in):
    return statistics.median(data_in["res_norm_kmal"])

def get_mad_res_kmal(data_in):
    #more robust to outliers
    return stats.median_abs_deviation(data_in["res_norm_kmal"])

# md1 = get_mean_res_kmal(data1)
# var1 = get_var_res_kmal(data1)
# md3 = get_mean_res_kmal(data3)
# var3 = get_var_res_kmal(data3)
# md4 = get_mean_res_kmal(data4)
# var4 = get_var_res_kmal(data4)
# md5 = get_mean_res_kmal(data5)
# var5 = get_var_res_kmal(data5)
# md6 = get_mean_res_kmal(data6)
# var6 = get_var_res_kmal(data6)


# mad1= get_mad_res_kmal(data1)
# mad3= get_mad_res_kmal(data3)
# mad4= get_mad_res_kmal(data4)
# mad5= get_mad_res_kmal(data5)
# mad6= get_mad_res_kmal(data6)

# med1= get_median_res_kmal(data1)
# med3= get_median_res_kmal(data3)
# med4= get_median_res_kmal(data4)
# med5= get_median_res_kmal(data5)
# med6= get_median_res_kmal(data6)

#%% main 

def keep_good_gc_shank(data_in, sub_nbr):
    delay_to_cut_r = 15
    delay_to_cut_l = 30
    nbr_steps_to_cut = 4    
    bound_res_kmal = 3
    max_val_shank = 7 #5 
    min_val_shank = 4
    min_slope = -0.1
    
    dict_data_to_cut = cut_data(data_in, delay_to_cut_r, delay_to_cut_l, nbr_steps_to_cut, sub_nbr)
    reset_idx(dict_data_to_cut)
    dict_cut = remove_bad_kmal_val(dict_data_to_cut, bound_res_kmal)
    # plot_sep(dict_cut)
    final_cut_data = remove_bad_shank_val(dict_cut, max_val_shank, min_val_shank, min_slope)

    # plot_sep(final_cut_data)
    
    return final_cut_data
    # return dict_cut

plt.figure()
plt.plot(data1["no_mc_shank_angle"])
plt.plot(data1["no_mc_kmal_angle"])
plt.plot(data1["res_norm_shank"], color = "fuchsia")
plt.plot(data1["res_norm_kmal"], color = "red")

c_data_1 = keep_good_gc_shank(data1, 1)
c_data_3 = keep_good_gc_shank(data3, 3)
c_data_4 = keep_good_gc_shank(data4, 4)
# c_data_5 = keep_good_gc_shank(data5, 5)
# c_data_6 = keep_good_gc_shank(data6, 6)

# plot_sep(c_data_1)
# plot_sep(c_data_3)
# plot_sep(c_data_4)

# all_good_shank_gc = {}
# all_good_shank_gc["sub1"] = c_data_1
# all_good_shank_gc["sub3"] = c_data_3
# all_good_shank_gc["sub4"] = c_data_4
# all_good_shank_gc["sub5"] = c_data_5
# all_good_shank_gc["sub6"] = c_data_6

# save_obj(all_good_shank_gc, "shank_good_gc")

