# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:15:07 2021

@author: ksdiv
look at residuals as metric to cut data 
"""

import pickle
import matplotlib.pyplot as plt
import statistics
from scipy.fft import fft, ifft, fftfreq

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
file1 = r"E:\ETHZ\mast_sem_IV\pdm\code02\test_shankdiff_functions_not_cut.pkl"
# file1c = r"S01_cut_Mocap_angles_res"
data1 = op_pickle(file1)

file2 = r"E:\ETHZ\mast_sem_IV\pdm\code02\test_functions_not_cut.pkl"
# file1c = r"S01_cut_Mocap_angles_res"
data2 = op_pickle(file2)

file_diff = r"E:\ETHZ\mast_sem_IV\pdm\code02\test_mod_getdirec_functions_not_cut.pkl"
data_diff = op_pickle(file_diff)

#%%
file1c = r"E:\ETHZ\mast_sem_IV\pdm\code02\DataMarc\S04_functions_not_cut.pkl"
data1c = op_pickle(file1c)

#%% plots
def plot_res_shank(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], label = "kma")
    plt.plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], label = "body part")
    plt.plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "res shank")
    plt.plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "res kma")
    # plt.plot(d1["t"][idx], d1["mc_kmal_angle"][idx], label = "kma")
    # plt.plot(d1["t"][idx], d1["mc_shank_angle"][idx], label = "body part")
    # plt.plot(d1["t"][idx], d1["res_norm_shank"][idx], label = "res shank")
    # plt.plot(d1["t"][idx], d1["res_norm_kmal"][idx], label = "res kma")
    plt.title("shank")
    plt.legend()
    return 

def plot_res_thigh(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmau_angle"][idx], label = "kma")
    plt.plot(d1["t"][idx], d1["no_mc_thigh_angle"][idx], label = "body part")
    plt.plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "res thigh")
    plt.plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "res kmau")
    
    # plt.plot(d1["t"][idx], d1["mc_kmau_angle"][idx], label = "kma")
    # plt.plot(d1["t"][idx], d1["mc_thigh_angle"][idx], label = "body part")
    # plt.plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "res thigh")
    # plt.plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "res kmau")
    plt.title("thigh")
    plt.legend()
    return 

plot_res_shank(data1c)
plot_res_shank(data_diff)
plot_res_thigh(data1)

#%%
idx = range(10, len(data1["t"]) - 10)
plt.figure()
plt.plot(data2["t"][idx], data1["no_mc_shank_angle"][idx], label = "data2")
plt.plot(data_diff["t"][idx], data2["no_mc_shank_angle"][idx], label = "data diff")
plt.legend()

#%%
idx = range(10, len(data1["t"]) - 10)
plt.figure()
plt.plot(data2["t"][idx], data1["no_mc_kmal_angle"][idx], label = "data2")
plt.plot(data_diff["t"][idx], data2["no_mc_kmal_angle"][idx], label = "data diff")
plt.legend()
plt.title("kmalow")


#%% fft

def get_fft_shank(d1):
    N = len(d1["t"])
    # shank_fft = fft(d1["no_mc_shank_angle"])
    y = d1["no_mc_shank_angle"].to_numpy()
    res_shank_fft = fft(y)
    xfft = fftfreq(N)
    
    fig, axes = plt.subplots(2)
    axes[0].plot(d1["t"], d1["no_mc_shank_angle"])
    axes[1].plot(abs(xfft), res_shank_fft)
    return res_shank_fft

# a = get_fft_shank(data1)
# p = get_fft_shank(data1c[32])




#%%
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
        plt.legend()
    return

