# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:20:43 2021

@author: ksdiv
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
        
subject = "SG"
fl = "FL1"
        
file1 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + subject + "/" + subject + "_" + fl + "/" + subject + "_" + fl + "_Isometric1.pkl"
data1 = op_pickle(file1)

file2 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + subject + "/" + subject + "_" + fl + "/" + subject + "_" + fl + "_Isometric2.pkl"
data2 = op_pickle(file2)

# file2 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SG/SG_FL5_1/SG_FL5_1_Isometric1.pkl"
# data2 = op_pickle(file2)

# file3 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_FL3/SA_FL3_Isometric1.pkl"
# data3 = op_pickle(file3)

# file4 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_FL3/SH_FL3_Isometric1.pkl"
# data4 = op_pickle(file4)

# file5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_FL5_2/SA_FL5_2_Concentric2.pkl"
# data5 = op_pickle(file5)

# file5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SF/INTERMEDIATE/SF_FUN_FL5_2.pkl"
# data5 = op_pickle(file5)

# file6 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SD/INTERMEDIATE/SD_FUN_FL5.pkl"
# data6 = op_pickle(file6)

# file6 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SB/INTERMEDIATE/SB_FUN_FL3.pkl"
# data6 = op_pickle(file6)

# file7 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SB/SB_FL3/SB_FL3_Isometric2.pkl"
# data7 = op_pickle(file7)

#%%

def plot_f(d_in):
    plt.figure()
    plt.plot(d_in["no_mc_shank_angle"], label = "shank")
    plt.plot(d_in["no_mc_kmal_angle"], label = "kmal")
    plt.plot(d_in["GyroCShank"], label = "shank gyro")
    plt.plot(d_in["no_mc_thigh_angle"], label = "thigh")
    plt.plot(d_in["no_mc_kmau_angle"], label = "kmau")
    plt.plot(d_in["GyroCThigh"], label = "thigh gyro")
    plt.plot(d_in["Mode"])
    plt.plot(d_in["current_sent"])
    # plt.plot(d_in["Force"])
    plt.legend()
    return

def plot_a(d_in):
    plt.figure()
    plt.plot(d_in["GyroCShank"], label = "shank gyro")
    plt.plot(d_in["AccelAShank"], label = "shank acc")
    plt.plot(d_in["GyroCThigh"], label = "thigh gyro")
    plt.plot(d_in["AccelAThigh"], label = "thigh acc")
    plt.plot(d_in["Mode"])
    plt.plot(d_in["current_sent"])
    # plt.plot(d_in["Force"])
    plt.legend()
    return
#%%
plot_f(data1)
plot_f(data2)
# plot_f(data3)
# plot_f(data4)
# plot_f(data5)
# plot_f(data6)
# plot_f(data7)

#%% 

plot_a(data1)
plot_a(data2)