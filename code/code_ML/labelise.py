# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:51:20 2021

@author: ksdiv
labelize gait cycle in function of wether the thigh angle show rotation or not 
fait ça sur données déjà coupées et mélangées - rend le meme dict mais avec une
colonne en plus dans le dataframe 

idée : lisse le kmau, puis regarde le nbr de pics - en fonction du nbr de pics 
va décider si il y a rotation ou pas 
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
from scipy.fft import fft, ifft, fftfreq

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long.pkl"
thigh_data = op_pickle(thigh_data_file)

#%% 

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

def plot_ind_thigh(d1):
    idx = range( len(d1["t"]))
    
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
    plt.legend()
    return 

# plot_sep_thigh(thigh_data["sub1"])
plot_sep_thigh(thigh_data["sub3"])
# plot_sep_thigh(thigh_data["sub4"])
plot_sep_thigh(thigh_data["sub5"])
# plot_sep_thigh(thigh_data["sub6"])


#%% 

e1 = thigh_data["sub4"][33]
e2 = thigh_data["sub6"][31]
e3 = thigh_data["sub5"][48]

plot_ind_thigh(e1)
plot_ind_thigh(e2)
plot_ind_thigh(e3)

#%% 

def get_fft_kmau(d1):
    N = len(d1["t"])
    # shank_fft = fft(d1["no_mc_shank_angle"])
    y = d1["no_mc_kmau_angle"].to_numpy()
    res_kmau_fft = fft(y)
    xfft = fftfreq(N)
    
    fig, axes = plt.subplots(2)
    axes[0].plot(d1["t"], d1["no_mc_kmau_angle"])
    axes[1].plot(abs(xfft), res_kmau_fft)
    return res_kmau_fft

e1_fft = get_fft_kmau(e1)
e2_fft = get_fft_kmau(e2)
e3_fft = get_fft_kmau(e3)

#%%
def get_grad_kmau(d1):
    grad = np.gradient(d1["no_mc_kmau_angle"])
    
    fig, axes = plt.subplots(2)
    axes[0].plot(d1["t"], d1["no_mc_kmau_angle"])
    axes[1].plot(d1["t"], grad)
    return grad

e1_grad = get_grad_kmau(e1)
e2_grad = get_grad_kmau(e2)
e3_grad = get_grad_kmau(e3)

#%%

def smooth_cut(d1):   
    win = scipy.signal.windows.hann(10)
    
    filtered = scipy.signal.convolve(d1["no_mc_kmau_angle"], win, mode = "same")/ sum(win)
    
    grad_filt = np.gradient(filtered)
    
    fig, axes = plt.subplots(3)
    axes[0].plot(d1["t"], d1["no_mc_kmau_angle"])
    axes[1].plot(d1["t"], filtered)    
    axes[2].plot(d1["t"], grad_filt)
    
    return filtered

e1_smoo = smooth_cut(e1)
e2_smoo = smooth_cut(e2)
e3_smoo = smooth_cut(e3)

#%% 

def peaks_and_plot(d1):
    p = scipy.signal.find_peaks(d1["no_mc_kmau_angle"])
    val_p = p[0][:]
    
    idx = range( len(d1["t"]))
    plt.figure()    
    plt.plot(d1["t"][idx], d1["no_mc_kmau_angle"][idx], label = "kma")
    plt.plot(d1["t"][val_p], d1["no_mc_kmau_angle"][val_p], "ro")
    return p

e1_peaks = peaks_and_plot(e1)
e2_peaks = peaks_and_plot(e2)
e3_peaks = peaks_and_plot(e3)

#%%

def peaks_and_plot_smoothed(d1):
    win = scipy.signal.windows.hann(10)
    
    filtered = scipy.signal.convolve(d1["no_mc_kmau_angle"], win, mode = "same")/ sum(win)
    
    p = scipy.signal.find_peaks(filtered)
    val_p = p[0][:]
    
    idx = range( len(d1["t"]))
    plt.figure()    
    plt.plot(d1["t"][idx], d1["no_mc_kmau_angle"])
    plt.plot(d1["t"][val_p], d1["no_mc_kmau_angle"][val_p], "ro")
    
    return p

e1_s_p = peaks_and_plot_smoothed(e1)
e2_s_p = peaks_and_plot_smoothed(e2)
e3_s_p = peaks_and_plot_smoothed(e3)

#%%

def peaks_and_plot_smoothed_dict(dict_in):
    win = scipy.signal.windows.hann(10)
    
    plt.figure()
    for key in dict_in:
        
        filt = scipy.signal.convolve(dict_in[key]["no_mc_kmau_angle"], win, mode = "same")/ sum(win)
        p = scipy.signal.find_peaks(filt)
        val_p = p[0][:]
        
        plt.plot(dict_in[key]["t"], dict_in[key]["no_mc_kmau_angle"])
        plt.plot(dict_in[key]["t"][val_p], dict_in[key]["no_mc_kmau_angle"][val_p], "ro")
        
    return

peaks_and_plot_smoothed_dict(thigh_data["sub1"])
peaks_and_plot_smoothed_dict(thigh_data["sub3"])
peaks_and_plot_smoothed_dict(thigh_data["sub4"])
peaks_and_plot_smoothed_dict(thigh_data["sub5"])
peaks_and_plot_smoothed_dict(thigh_data["sub6"])
