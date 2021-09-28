# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:37:00 2021

@author: ksdiv
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
from scipy import interpolate as itp
import pandas as pd
import os
import glob 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


#%% Naming

# change this
subject = "SE"

# SB doesn't have FL5 ! (fichier existe mais va pas fonctioner )

# directory subject (where to create new file)
# myosuit 
dir_name_sub = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\\" + subject
# mocap
dir_name_sub = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MOCAP\\" + subject

list_fl_files = glob.glob(dir_name_sub + "\*.pkl")

#%%

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def save_all_bits_sep(dict_data, dirName):
    for key in dict_data:
        tmp_name = dirName + "_" + key
        save_obj(dict_data[key], tmp_name)
    return

## myosuit
# for fl_file in list_fl_files:
#     temp_fl = fl_file[46:]
#     data1 = op_pickle(fl_file)
    
#     current_sent = data1["CurrentRightMotor"]
#     current_received = data1["ActualCurrentRightMotor"]
#     AlphaShank = data1["IMURightShankKalmanAngle"]
#     AlphaThigh = data1["IMURightThighKalmanAngle"]
#     HallSensor = data1["HallRightMotor"]
    
#     plt.figure()
#     plt.plot(HallSensor, label = "hall")
#     plt.plot(AlphaShank/80, label = "shank")
#     plt.plot(current_sent, label = "current_sent")
#     plt.plot(current_received, label = "current received")
#     plt.title(temp_fl)
#     plt.legend()

# mocap
pathMoCap = list_fl_files[1]

file = open(pathMoCap,'rb')
MoCapData = pickle.load(file)
t = MoCapData["time"]
labelsMoCap = MoCapData["labels"]
markerpos = MoCapData["points_array"] 
vgrf = MoCapData["vgrf"]

Ftot_vert = vgrf[2,:] + vgrf[8,:]

plt.figure()
plt.plot(-vgrf[2,:], label = "2")
plt.plot(-vgrf[8,:], label = "8")
plt.plot(Ftot_vert, label = "tot")
plt.legend()
