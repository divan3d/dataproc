# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:14:08 2021

@author: ksdiv

save gait cycles so that they have equal size
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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


#%% Naming

# change this
subject = "SD"

# directory subject 
dir_name_sub = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject
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

def save_in_new_file(dirName, dict_data, name_file):  
    os.chdir(dir_name_sub)
            
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")    
        
    os.chdir(dirName)
    # print("Current working directory: {0}".format(os.getcwd()))
    save_obj(dict_data, name_file) 
    return
    
    
#%% dir 

# directory
dir_name_cut = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_CUT_EGC"

# new file name - a voir 
new_file_shank = dir_name_sub + "\\" + subject + "_EQ_GC_shank"
new_file_thigh = dir_name_sub + "\\" + subject + "_EQ_GC_thigh"

# shank file
shank_dir = dir_name_cut + "/shank"
shank_list = os.listdir(shank_dir)

# thigh file 
thigh_dir = dir_name_cut + "/thigh"
thigh_list = os.listdir(thigh_dir)

#%% functions equal time 

def get_equal_time(s_in, length):
    inter = itp.interp1d(range(len(s_in)), s_in)
    out = inter(np.linspace(0, len(s_in)-1, length))
    return out 

def dict_equal_time(dict_in, length):
    eq_d1 = {}
    for key in dict_in:
        temp =  {}
        temp["t"] = get_equal_time(dict_in[key]["t"], length)
        temp["no_mc_thigh_angle"] = get_equal_time(dict_in[key]["no_mc_thigh_angle"], length)
        temp["no_mc_kmau_angle"] = get_equal_time(dict_in[key]["no_mc_kmau_angle"], length)
        temp["no_mc_shank_angle"] = get_equal_time(dict_in[key]["no_mc_shank_angle"], length)
        temp["no_mc_kmal_angle"] = get_equal_time(dict_in[key]["no_mc_kmal_angle"], length)
        temp["vgrf"] = get_equal_time(dict_in[key]["vgrf"], length)
        temp["vgrf1"] = get_equal_time(dict_in[key]["vgrf1"], length)
        temp["vgrf2"] = get_equal_time(dict_in[key]["vgrf2"], length)
        temp["current_sent"] = get_equal_time(dict_in[key]["current_sent"], length)
        temp["current_read"] = get_equal_time(dict_in[key]["current_read"], length)
        temp["L_leg"] = get_equal_time(dict_in[key]["L_leg"], length)
        temp["R_leg"] = get_equal_time(dict_in[key]["R_leg"], length)
        temp["AlphaShank"] = get_equal_time(dict_in[key]["AlphaShank"], length)
        temp["AlphaThigh"] = get_equal_time(dict_in[key]["AlphaThigh"], length)
        temp["Mode"] = get_equal_time(dict_in[key]["Mode"], length)
        temp["Force"] = get_equal_time(dict_in[key]["Force"], length)
        temp["ForceLevel"] = get_equal_time(dict_in[key]["ForceLevel"], length)
        temp["GyroAThigh"] = get_equal_time(dict_in[key]["GyroAThigh"], length)
        temp["GyroAShank"] = get_equal_time(dict_in[key]["GyroAShank"], length)
        temp["GyroBThigh"] = get_equal_time(dict_in[key]["GyroBThigh"], length)
        temp["GyroBShank"] = get_equal_time(dict_in[key]["GyroBShank"], length)
        temp["GyroCThigh"] = get_equal_time(dict_in[key]["GyroCThigh"], length)
        temp["GyroCShank"] = get_equal_time(dict_in[key]["GyroCShank"], length)
        temp["AccelAThigh"] = get_equal_time(dict_in[key]["AccelAThigh"], length)
        temp["AccelAShank"] = get_equal_time(dict_in[key]["AccelAShank"], length)
        temp["AccelBThigh"] = get_equal_time(dict_in[key]["AccelBThigh"], length)
        temp["AccelBShank"] = get_equal_time(dict_in[key]["AccelBShank"], length)
        temp["AccelCThigh"] = get_equal_time(dict_in[key]["AccelCThigh"], length)
        temp["AccelCShank"] = get_equal_time(dict_in[key]["AccelCShank"], length)
        temp["HallSensor"] = get_equal_time(dict_in[key]["HallSensor"], length)
        temp["InitialLength"] = np.ones(length)*len(dict_in[key])
        eq_d1[key] = pd.DataFrame.from_dict(temp)
    return eq_d1

#%% start with shank 

length_gc = 170

for f_name in shank_list:
    file_now = op_pickle(shank_dir + "/" + f_name) 
    temp = dict_equal_time(file_now, length_gc)
    save_in_new_file(new_file_shank, temp, f_name)


#%%  thigh 

for f_name in thigh_list:
    file_now = op_pickle(thigh_dir + "/" + f_name) 
    tempt = dict_equal_time(file_now, length_gc)
    save_in_new_file(new_file_thigh, tempt, f_name)



