# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:53:51 2021

@author: ksdiv

cut the isometric data to have same number of points so that can compare 
between experiments 
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
subject = "SH"

# SB doesn't have FL5 ! (fichier existe mais va pas fonctioner )

# directory subject (where to create new file)
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
    
#%% dir 

# directory - where to retrieve file 
# ce qui est un peu relou - nom change 
dir_name_cut = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\\" + subject + "_CUT_EGC"

# new file name - a voir 
new_file_iso = dir_name_sub + "\\" + subject + "_EQ_iso"


# get FL files for subject 
list_fl_files = glob.glob(dir_name_sub + "\\" + subject + "_FL*")



#%% separate

# a partir du mode change, prend un certain nombre de points - prendre en compte que pr iso1 et 2, le mode commence a un autre endroit 

def find_mode_change(d_in):
    """
    function to separate the different sections of one trial (1/2). Does this by 
    finding the changes in Mode from MyoSuit data.

    Parameters
    ----------
    d_in : dataframe : whole trial 

    Returns
    -------
    idx_mode_change : list : contains indices of changes in Mode
    """
    d_in["Mode"] = d_in["Mode"].round()
    mode_change = d_in["Mode"].shift() != d_in["Mode"]
    idx_mode_change = mode_change.index[mode_change == True].tolist()
    
    #change to adjust to indexing
    idx_to_cut = [x - idx_mode_change[0] for x in idx_mode_change]
    
    # print("nbr of mode changes = %i" %len(idx_mode_change))
    
    # plt.figure()
    # plt.plot(d_in["Mode"])
    # plt.plot(idx_mode_change, d_in["Mode"][idx_mode_change],"o")
    
    return idx_to_cut

def separate_iso1(d_in, idx_to_cut):
    #normal
    offset_on = 100
    offset_off = 80
    
    
    on_1 = d_in[idx_to_cut[1] : (idx_to_cut[1] + offset_on)]
    on_1 = on_1.reset_index()
    off_1 = d_in[idx_to_cut[2] : (idx_to_cut[2] + offset_off)]
    off_1 = off_1.reset_index()
    on_2 = d_in[idx_to_cut[3] : (idx_to_cut[3] + offset_on)]
    on_2 = on_2.reset_index()
    off_2 = d_in[idx_to_cut[4] : (idx_to_cut[4] + offset_off)]
    off_2 = off_2.reset_index()
    
    dict_iso1 = {"iso1_on_1" : on_1, "iso1_off_1" : off_1, "iso1_on_2" : on_2, "iso1_off_2" : off_2}
    return dict_iso1

def separate_iso2(d_in, idx_to_cut):
    #normal
    offset_on = 100
    offset_off = 80
        
    on_1 = d_in[idx_to_cut[0] : (idx_to_cut[0] + offset_on)]
    on_1 = on_1.reset_index()
    off_1 = d_in[idx_to_cut[1] : (idx_to_cut[1] + offset_off)]
    off_1 = off_1.reset_index()
    on_2 = d_in[idx_to_cut[2] : (idx_to_cut[2] + offset_on)]
    on_2 = on_2.reset_index()
    off_2 = d_in[idx_to_cut[3] : (idx_to_cut[3] + offset_off)]
    off_2 = off_2.reset_index()
    
    dict_iso1 = {"iso2_on_1" : on_1, "iso2_off_1" : off_1, "iso2_on_2" : on_2, "iso2_off_2" : off_2}
    return dict_iso1

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


#%% test 

# f_in = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_FL1/SA_FL1_Isometric1.pkl"
# d_i1 = op_pickle(f_in)

# f_i2 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_FL1/SA_FL1_Isometric2.pkl"
# d_i2 = op_pickle(f_i2)

# find_mode_change(d_i1)
# find_mode_change(d_i2)

# #%%

# i1, ic1 = find_mode_change(d_i1)
# sep = separate_iso1(d_i1, ic1)
# plot_f(sep["iso1_on_1"])


#%%


for fl_file in list_fl_files:
    # attention au chiffre ici !! 
    temp_fl = fl_file[46:]
    list_iso_files = glob.glob(fl_file + "\\" +  subject + "_" + temp_fl + "_Isometric*")
    for iso_file in list_iso_files:
        iso_nbr = int(iso_file[-5])
        if iso_nbr == 1:
            d_iso_1 = op_pickle(iso_file)
            places_to_cut_1 = find_mode_change(d_iso_1)
            iso1_dict = separate_iso1(d_iso_1, places_to_cut_1)
            new_name = subject + "_" + temp_fl + "_cut_Isometric1.pkl"
            save_in_new_file(new_file_iso, iso1_dict, new_name)
        if iso_nbr == 2:
            d_iso_2 = op_pickle(iso_file)
            places_to_cut_2 = find_mode_change(d_iso_2)
            iso2_dict = separate_iso2(d_iso_2, places_to_cut_2)
            new_name = subject + "_" + temp_fl + "_cut_Isometric2.pkl"
            save_in_new_file(new_file_iso, iso2_dict, new_name)
            
    # veut faire pr que ça rende le nombre de dict qu'il y a d'experience isometrique 
    
    
