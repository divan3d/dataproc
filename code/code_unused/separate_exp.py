# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:19:46 2021

@author: ksdiv
separates experiment into smaller subsets 
"""

import pickle
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import statistics
import os

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
# file1 = r"E:\ETHZ\mast_sem_IV\pdm\code02\test_shankdiff_functions_not_cut.pkl"
# # file1c = r"S01_cut_Mocap_angles_res"
# data1 = op_pickle(file1)

file1 = r"E:\ETHZ\mast_sem_IV\pdm\code02\SA\SA_200721_FUN_FL1_MS.pkl"
# file1c = r"S01_cut_Mocap_angles_res"
data1 = op_pickle(file1)

file3 = r"E:\ETHZ\mast_sem_IV\pdm\code02\SA\SA_200721_FUN_FL3_MS.pkl"
# file1c = r"S01_cut_Mocap_angles_res"
data3 = op_pickle(file3)

# file1c = r"E:\ETHZ\mast_sem_IV\pdm\code02\DataMarc\S04_functions_not_cut.pkl"
# data1c = op_pickle(file1c)
#%%
def plot_d(d1):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], label = "kmal")
    plt.plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], label = "shank")
    # plt.plot(d1["t"][idx], d1["no_mc_kmau_angle"][idx], label = "kmau")
    # plt.plot(d1["t"][idx], d1["no_mc_thigh_angle"][idx], label = "thigh")
    plt.title("extracted angles")
    plt.legend()
    return 

# plot_d(data1)

# plot_d(data1c)
#%%
mm = scipy.signal.find_peaks(data1["no_mc_shank_angle"], height=(5))

def plot_peaks(d1, pea):
    idx = range(10, len(d1["t"]) - 10)
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["no_mc_kmal_angle"][idx], label = "kmal")
    plt.plot(d1["t"][idx], d1["no_mc_shank_angle"][idx], label = "shank")
    plt.plot(d1["t"][pea], d1["no_mc_shank_angle"][pea], "o", label = "peaks")
    plt.title("extracted angles")
    plt.legend()
    return 

# plot_peaks(data1, mm[0])

# plt.figure()
# plt.plot(range(len(mm[0])),mm[0])



#%% 

# plt.figure()
# plt.plot(data1["Mode"])
# plt.plot(data1["no_mc_shank_angle"])

# plt.figure()
# plt.plot(data3["Mode"])
    
#%% get index where mode changes value 



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
    print("nbr of mode changes = %i" %len(idx_mode_change))
    return idx_mode_change

def separate_data(d_in, idx_mode_change):
    """
    function to separate the different sections of one trial (2/2). Separates
    the input dataframe in relation to mode value and the structure of the trial.

    Parameters
    ----------
    d_in : dataframe : whole trial 
    idx_mode_change : list : output of find_mode_change, indices of mode change 

    Returns
    -------
    dict_data : dict : contains the separated trials

    """
    transp = d_in[idx_mode_change[0]: idx_mode_change[1]]
    iso1 = d_in[idx_mode_change[1]: idx_mode_change[5]]
    concentric1 = d_in[idx_mode_change[5]: idx_mode_change[6]]
    lag_val = idx_mode_change[8] - idx_mode_change[7]
    iso2 = d_in[idx_mode_change[6]: (idx_mode_change[9] + lag_val)]
    concentric2 = d_in[idx_mode_change[9]:]
    dict_data =  {"Transparent" : transp, "Isometric1" : iso1, "Concentric1" : concentric1,
                 "Isometric2": iso2, "Concentric2" : concentric2}
    return dict_data



def check_mode(dict_data):    
    """
    prints mean value of mode for each separated trial. 

    Parameters
    ----------
    dict_data : dict : contains separated trials

    Returns
    -------
    None.

    """
    print(" checking mode values of separated trials : ")
    for key in dict_data:
        mode_check = statistics.mean(dict_data[key]["Mode"])
        print( key)
        print("mean mode : %f" % mode_check)
    return

# indices = find_mode_change(data3)


# plt.figure()
# plt.plot(data3["Mode"])
# plt.plot(indices, data3["Mode"][indices], "o")

# dd = separate_data(data3, indices)

# #%%

# i1 = find_mode_change(data1)
# d1 = separate_data(data1, i1)

# plt.figure()
# plt.plot(data1["Mode"])
# plt.plot(i1, data1["Mode"][i1], "o")

# # plt.figure()
# # plt.plot(d1["Isometric1"]["Mode"])
# # plt.plot(d1["Isometric1"]["no_mc_shank_angle"])
# #%%
# plt.figure()
# plt.plot(d1["Isometric2"]["Mode"])
# plt.plot(d1["Isometric2"]["no_mc_shank_angle"])

# plt.figure()
# plt.plot(d1["Concentric2"]["Mode"])
# plt.plot(d1["Isometric2"]["no_mc_shank_angle"])


# plt.figure()
# plt.plot(dd["Isometric2"]["Mode"])
# plt.plot(dd["Isometric2"]["no_mc_shank_angle"])

# plt.figure()
# plt.plot(dd["Isometric2"]["no_mc_shank_angle"])
# plt.plot(dd["Concentric2"]["Mode"])
#%% create new directory, with files for each part of trial 


def save_all_bits_sep(dict_data, dirName):
    for key in dict_data:
        tmp_name = dirName + "_" + key
        save_obj(dict_data[key], tmp_name)
    return

def separate_trial(data_in, dirName, save_or_not): 
    """
    Separates the trial at a certain force level into the transparent, isometric
    and concentric parts and saves the separated trials into new directory. 
    Prints mode mean values to check. 

    Parameters
    ----------
    dict_in : dataframe : whole trial, offset removed 
    dirName : string : name of new directory, form : S#_FL#
    save_or_not : bool : wether to save the separated trials or not

    Returns
    -------
    None.

    """
    
    #separate
    mode_change_idx = find_mode_change(data_in)
    sep_data = separate_data(data_in, mode_change_idx)
    check_mode(sep_data)
    
    # save
    if save_or_not :
        os.chdir(r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test")
                
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")    
            
        os.chdir(dirName)
        print("Current working directory: {0}".format(os.getcwd()))
    
        save_all_bits_sep(sep_data, dirName) 
    return


    