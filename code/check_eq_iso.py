# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:15:17 2021

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
        
        
        
f11 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_iso/SH_FL1_cut_Isometric1.pkl"
d11 = op_pickle(f11) 
f12 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_iso/SH_FL1_cut_Isometric2.pkl"
d12 = op_pickle(f12)

fc32 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_EQ_iso/SC_FL3_cut_Isometric2.pkl"
dc32 = op_pickle(fc32)


def plot_a(d_in):
    
    plt.plot(d_in["GyroCShank"], label = "shank gyro")
    plt.plot(d_in["AccelAShank"], label = "shank acc")
    plt.plot(d_in["GyroCThigh"], label = "thigh gyro")
    plt.plot(d_in["AccelAThigh"], label = "thigh acc")
    plt.plot(d_in["Mode"])
    plt.plot(d_in["current_sent"])
    # plt.plot(d_in["Force"])
    plt.legend()
    return


plt.figure()
plot_a(d11["iso1_on_1"])
plot_a(d11["iso1_on_2"])
plot_a(d12["iso2_on_1"])
plot_a(d12["iso2_on_2"])
plot_a(dc32["iso2_on_1"])
plot_a(dc32["iso2_on_2"])