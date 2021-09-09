# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:22:34 2021

@author: ksdiv
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
        
file1 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SG/SG_FL5_2/SG_FL5_2_Concentric2.pkl"
data1 = op_pickle(file1)

#%%

# def dyn_places_to_cut(d1):
#     """
#     finds maximum of vgrf -- corresponds to heel strike for each leg

#     Parameters
#     ----------
#     d1 : dataframe containing dynamic trial

#     Returns
#     -------
#     list_of_max : list, contains indices of max values, 0 and last value added 
#     to help cut the data later

#     """
#     # max2 = scipy.signal.argrelextrema(d1["vgrf"].values, np.greater_equal, order = 100)
#     p, properties = scipy.signal.find_peaks(d1["vgrf"].values, height=500, distance = 40)
#     # size_data = len(d1)
#     # list_of_max =  max2[0]
#     # list_of_max = np.append(list_of_max, size_data)
#     # list_of_max = np.append(0, list_of_max)
#     return p

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
    # max2 = scipy.signal.argrelextrema(d1["no_mc_shank_angle"].values, np.greater_equal, order = 100)
    p, properties = scipy.signal.find_peaks(d1["no_mc_shank_angle"].values, height=15, distance = 40)
    size_data = len(d1)
    # list_of_max =  max2[0]
    # list_of_max = np.append(list_of_max, size_data)
    # list_of_max = np.append(0, list_of_max)
    return p

def plot_Fig1(dict_in):
    plt.figure()
    # plt.plot(dict_in["t"], dict_in["vgrf"] - 500 , label = "vgrf")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.plot(dict_in["t"], dict_in["AlphaShank"] - 80, label = "shank myosuit")
    plt.plot(dict_in["t"], dict_in["R_leg"] , label = "stance R leg")
    plt.plot(dict_in["t"], dict_in["L_leg"] , label = "stance L leg")
    plt.legend()
    plt.title("FIG 1: vgrf, shank angles")
    return 

lmax = dyn_places_to_cut(data1)

data1 = data1.reset_index()

plot_Fig1(data1)
plt.plot(data1["t"][lmax], data1["no_mc_shank_angle"][lmax], marker = "o")

#%%

# p, properties = scipy.signal.find_peaks(data1["vgrf"].values, height=400)