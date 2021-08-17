# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:15:47 2021

@author: ksdiv

see if can retrieve data from static case 
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import statistics
import sklearn.metrics
import scipy.signal

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
#%%
file_s1 = r"E:\ETHZ\mast_sem_IV\pdm\marc\code\marcwirththesis\S01_Static_Postprocessed_Mocap_angles.pkl"
data_1s = op_pickle(file_s1)
file_s3 = r"E:\ETHZ\mast_sem_IV\pdm\marc\code\marcwirththesis\S03_Static_Postprocessed_Mocap_angles.pkl"
data_3s = op_pickle(file_s3)
file_s4 = r"E:\ETHZ\mast_sem_IV\pdm\marc\code\marcwirththesis\S04_Static_Postprocessed_Mocap_angles.pkl"
data_4s = op_pickle(file_s4)
file_s5 = r"E:\ETHZ\mast_sem_IV\pdm\marc\code\marcwirththesis\S05_Static_Postprocessed_Mocap_angles.pkl"
data_5s = op_pickle(file_s5)
file_static =   r"E:\ETHZ\mast_sem_IV\pdm\marc\code\marcwirththesis\S06_Static_Postprocessed_Mocap_angles.pkl"
data_6s = op_pickle(file_static)

#%% cut data  

def indices_to_cut_static(d1):
    """
    finds indices to cut static data and separate each position 

    Parameters
    ----------
    d1 : pd dataframe : static data of 1 subject

    Returns
    -------
    left_peak : array : indices of the left sides of 0 plateau in current
    right_peak : array : indices of the right sides of 0 plateau in current   
    
    """
    rd_current = round(d1["current_sent"])
    _, properties = scipy.signal.find_peaks(rd_current.values, height=0, plateau_size = 500)
    left_peak = properties["left_edges"] + 15
    right_peak = properties["right_edges"]
    left_peak = np.append(left_peak, len(d1["current_sent"])-1)
    right_peak = np.append(0, right_peak)
    return left_peak, right_peak

def cut_each_pos(data_in):
    """
    cuts static data, creates dict with each position separated 

    Parameters
    ----------
    data_in : pd dataframe : static data of 1 subject

    Returns
    -------
    temp_dict : dict : key 0 to 4, corresponds to each position

    """
    l_val, r_val = indices_to_cut_static(data_in)    
    temp_dict = {}
    counter = 0
    for idx in range(0, len(l_val), 2):
        temp_dict[counter] = data_in.iloc[r_val[idx]:l_val[idx + 1 ]]
        counter = counter + 1
    return temp_dict

# data_1_cut = cut_each_pos(data_1s)

#%% remove offset 

def remove_offset(data_in):
    """
    removes offset between thigh and kmau angle, creates new column in dataframe

    Parameters
    ----------
    data_in : pd dataframe : static data of 1 subject/ 1 pos for 1 subject

    Returns
    -------
    None.

    """
    mean1 = statistics.mean(data_in["mc_thigh_angle"][3:-3])
    mean2 = statistics.mean(data_in["mc_kmau_angle"][3:-3])
    
    #ça envoie un warning 
    # pour la partie data_in["no_mc_##_angle] = bla
    # "a value is trying to be set on a copy of a slice from a DataFrame" 
    data_in["no_mc_thigh_angle"] = data_in["mc_thigh_angle"] - mean1
    data_in["no_mc_kmau_angle"] = data_in["mc_kmau_angle"] - mean2
    
    return

# remove_offset(data_1s)

def get_removed_offset_cut_data(data_cut):
    for key in data_cut:
        remove_offset(data_cut[key])
    return

# get_removed_offset_cut_data(data_1_cut)

#%% get mean avrg estimate 

def mae_without_outlyer(val1, val2):
    """
    gets mean average estimate btwn input, doesn't take into account if diff btwn
    values is larger than 10 (since good chance that's due to error)

    Parameters
    ----------
    val1 : pd series : KMAU or thigh angles (or other)
    val2 : pd series : KMAU or thigh angles (or other)

    Returns
    -------
    err : float64: mean avrg estimate btwn input values 

    """
    col1 = []
    col2 = []
    first_index = val1.index[0]
    for i in range(first_index, first_index + len(val1)):
        if abs(val1[i] - val2[i]) < 10:
            col1.append(val1[i])
            col2.append(val2[i])
    err = sklearn.metrics.mean_absolute_error(col1,col2)    
    return err

def get_mae_pos(pos, data_cut):
    temp = (mae_without_outlyer(data_cut[pos]["no_mc_thigh_angle"], data_cut[pos]["no_mc_kmau_angle"]))
    return temp

def get_error_each_pos(data_cut):
    temp = []
    temp.append(get_mae_pos(0, data_cut))
    temp.append(get_mae_pos(1,data_cut))
    temp.append(get_mae_pos(2, data_cut))
    temp.append(get_mae_pos(3,data_cut))
    temp.append(get_mae_pos(4, data_cut))
    return temp

# d1_err = get_error_each_pos(data_1_cut)

#%% plots 

def plot_check(data_in, left_peak, right_peak):
    " plot to check where the data is cut"
    idx = range(10, len(data_in["t"]) - 10)
    plt.figure()
    plt.plot(data_in["t"][idx], data_in["current_sent"][idx], color = "green", label = "current sent")
    plt.plot(data_in["t"][left_peak],  data_in["current_sent"][left_peak], "ro", label = "left")
    plt.plot(data_in["t"][right_peak],  data_in["current_sent"][right_peak], "bo", label = "right")
    return

l1, r1 = indices_to_cut_static(data_1s)
plot_check(data_1s, l1, r1)

def plot_force_mocap_angles_thigh(data_in):
    first_idx = data_in["t"].index[0]
    idx = range(first_idx, first_idx + len(data_in["t"]) - 10)
    plt.figure()      
    # plt.plot(data_in["t"][idx], data_in["mc_kmal_angle"][idx], color = "lightsalmon", label = "kmal")
    # plt.plot(data_in["t"][idx], data_in["mc_shank_angle"][idx], color = "lightsteelblue", label = "shank")
    # plt.plot(data_in["t"][idx], data_in["mc_kmau_angle"][idx], color = "lightsalmon", label = "kma")
    # plt.plot(data_in["t"][idx], data_in["mc_thigh_angle"][idx], color = "lightsteelblue", label = "thigh")
    plt.plot(data_in["t"][idx], data_in["mc_kmau_angle"][idx]-85, color = "lightsalmon", label = "kma")
    plt.plot(data_in["t"][idx], data_in["mc_thigh_angle"][idx]-80, color = "lightsteelblue", label = "thigh")
    # plt.plot(data_in["t"][idx], data_in["force"][idx], color = "springgreen", label = "force")
    plt.plot(data_in["t"][idx], data_in["current_sent"][idx], color = "green", label = "current sent")
    plt.legend()
    return

# plot_force_mocap_angles_thigh(data_1s) 




#%% main 
    
def separate_and_remove_offset(data_in):
    data_cut_return = cut_each_pos(data_in)
    get_removed_offset_cut_data(data_cut_return)
    return data_cut_return

# S3 pas utilisable 

data_1_cut = separate_and_remove_offset(data_1s)
data_3_cut = separate_and_remove_offset(data_3s)
data_4_cut = separate_and_remove_offset(data_4s)
data_5_cut = separate_and_remove_offset(data_5s)
data_6_cut = separate_and_remove_offset(data_6s)

data_1_err = get_error_each_pos(data_1_cut)
# data_3_err = get_error_each_pos(data_3_cut)
# data_4_err = get_error_each_pos(data_4_cut)
# data_5_err = get_error_each_pos(data_5_cut)
# data_6_err = get_error_each_pos(data_6_cut)
    

plot_force_mocap_angles_thigh(data_1_cut[0])
plot_force_mocap_angles_thigh(data_3_cut[0])
plot_force_mocap_angles_thigh(data_4_cut[0])
plot_force_mocap_angles_thigh(data_5_cut[0])
plot_force_mocap_angles_thigh(data_6_cut[0])



#%% regarde que l'exp 0 (pos 1, avec la jambe vers le bas)
# enleve offset a cet endroit, et trouve difference a chaque endroit ou il y a un courant appliqué
# stocke séparement tout les peaks (10 en tout) -- puis vera apres 

def cut_pos_events(dataframe_in):
    # faut couper/séparer en 12 evenements en tout - donc coupe a 11 endroits
    
    win= scipy.signal.windows.hann(50)
    filt = scipy.signal.convolve(dataframe_in["current_sent"], win, mode = "same")
    peaks, _ = scipy.signal.find_peaks(filt, prominence=(20))
    rd_current = round(dataframe_in["current_sent"]).to_numpy()
    rd_current = np.append(-5, rd_current)
    _, properties = scipy.signal.find_peaks(rd_current, height=0, plateau_size = 500)
    right_peak = properties["right_edges"]
    # plt.figure()
    # plt.plot(dataframe_in["t"], dataframe_in["current_sent"], color = "green", label = "current sent")
    # plt.plot(dataframe_in["t"], filt, color = "red", label = "current sent")
    # plt.plot(dataframe_in["t"][peaks], dataframe_in["current_sent"][peaks], "x")    
    # plt.plot(dataframe_in["t"][right_peak],  dataframe_in["current_sent"][right_peak], "bo", label = "right")
    
    cut = np.empty(1)
    cut = np.append(cut,right_peak)
    cut = np.append(cut,peaks)
    cut = cut[1:]
    cut = np.sort(cut).astype(int)
    return cut




def remove_offset_e1(dataframe_in, cut_pos1):
    mean1 = statistics.mean(dataframe_in["mc_thigh_angle"][:cut_pos1[0]])    
    mean2 = statistics.mean(dataframe_in["mc_kmau_angle"][:cut_pos1[0]])
    
    dataframe_in["fno_mc_thigh_angle"] = dataframe_in["mc_thigh_angle"] - mean1
    dataframe_in["fno_mc_kmau_angle"] = dataframe_in["mc_kmau_angle"] - mean2
    
    return

    

def remove_offset_e1_3c(dataframe_in, cut_pos1):
    mean1 = statistics.mean(dataframe_in["mc_thigh_angle"][cut_pos1[1]:cut_pos1[1]+400])    
    mean2 = statistics.mean(dataframe_in["mc_kmau_angle"][cut_pos1[1]:cut_pos1[1]+400])
    
    dataframe_in["fno_mc_thigh_angle"] = dataframe_in["mc_thigh_angle"] - mean1
    dataframe_in["fno_mc_kmau_angle"] = dataframe_in["mc_kmau_angle"] - mean2
    
    return 
#%%


def plot_pos1(dataframe_in, subnbr):
    plt.figure()
    plt.plot(dataframe_in["t"], dataframe_in["fno_mc_thigh_angle"], color = "green", label = "current sent")
    plt.plot(dataframe_in["t"], dataframe_in["fno_mc_kmau_angle"], color = "red", label = "current sent")
    plt.plot(dataframe_in["t"], dataframe_in["current_sent"], color = "lightsteelblue", label = "current sent")
    plt.title("sub " + str(subnbr))
    return

#%%

def get_diff_at_peaks(dataframe_in, cut_pos1):
    #enleve offset entre thigh et kmau
    # regarde juste les pics de courant 
    # trouve diff entre thigh et kmau a cet instant 
    error_array = []
    cut_pos1 = np.append(cut_pos1,len(dataframe_in["t"]))
    for u in range(len(cut_pos1)-1):
        error_array.append(sklearn.metrics.mean_absolute_error(
            dataframe_in["fno_mc_thigh_angle"][cut_pos1[u]:cut_pos1[u+1]], dataframe_in["fno_mc_kmau_angle"][cut_pos1[u]:cut_pos1[u+1]]))
    
    return error_array

#%%
cut_d1 = cut_pos_events(data_1_cut[0])
remove_offset_e1_3c(data_1_cut[0], cut_d1)
plot_pos1(data_1_cut[0], 1)
er_d1 = get_diff_at_peaks(data_1_cut[0], cut_d1)
er_ut_d1 = statistics.mean((er_d1[1],er_d1[2]))

cut_d3 = cut_pos_events(data_3_cut[0])
remove_offset_e1_3c(data_3_cut[0], cut_d3)
plot_pos1(data_3_cut[0], 3)
er_d3 = get_diff_at_peaks(data_3_cut[0], cut_d3)
er_ut_d3 = statistics.mean((er_d3[1],er_d3[2]))

cut_d4 = cut_pos_events(data_4_cut[0])
remove_offset_e1_3c(data_4_cut[0], cut_d4)
plot_pos1(data_4_cut[0], 4)
er_d4 = get_diff_at_peaks(data_4_cut[0], cut_d4)
er_ut_d4 = statistics.mean((er_d4[1],er_d4[2]))

cut_d5 = cut_pos_events(data_5_cut[0])
remove_offset_e1_3c(data_5_cut[0], cut_d5)
plot_pos1(data_5_cut[0], 5)
er_d5 = get_diff_at_peaks(data_5_cut[0], cut_d5)
er_ut_d5 = statistics.mean((er_d5[1],er_d5[2]))

cut_d6 = cut_pos_events(data_6_cut[0])
remove_offset_e1_3c(data_6_cut[0], cut_d6)
plot_pos1(data_6_cut[0], 6)
er_d6 = get_diff_at_peaks(data_6_cut[0], cut_d6)
er_ut_d6 = statistics.mean((er_d6[1],er_d6[2]))



