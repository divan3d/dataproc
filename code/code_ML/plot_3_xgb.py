# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:16:57 2021

@author: ksdiv
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy import random
# from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb 

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank.pkl"
# d = op_pickle(f)

# subAd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SA.pkl"
# subA = op_pickle(subAd)

# subDd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SD.pkl"
# subD = op_pickle(subDd)

# subEd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SE.pkl"
# subE = op_pickle(subEd)

# subFd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SF.pkl"
# subF = op_pickle(subFd)

# subGd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SG.pkl"
# subG = op_pickle(subGd)

# subHd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SH.pkl"
# subH = op_pickle(subHd)

#%%

da11 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_knee/SA_FL1_cut_knee_1.pkl"
fa11 = op_pickle(da11)

da12 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_knee/SA_FL1_cut_knee_2.pkl"
fa12 = op_pickle(da12)

da31 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_knee/SA_FL3_cut_knee_1.pkl"
fa31 = op_pickle(da31)

da32 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_knee/SA_FL3_cut_knee_2.pkl"
fa32 = op_pickle(da32)

da51 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_knee/SA_FL5_2_cut_knee_1.pkl"
fa51 = op_pickle(da51)

da52 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_knee/SA_FL5_2_cut_knee_2.pkl"
fa52 = op_pickle(da52)

dc51 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_EQ_GC_knee/SC_FL5_cut_knee_1.pkl"
fc51 = op_pickle(dc51)

dc52 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_EQ_GC_knee/SC_FL5_cut_knee_2.pkl"
fc52 = op_pickle(dc52)

de51 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SE/SE_EQ_GC_knee/SE_FL5_1_cut_knee_1.pkl"
fe51 = op_pickle(de51)

dh52 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_GC_knee/SH_FL5_cut_knee_2.pkl"
fh52 = op_pickle(dh52)

#%%
# dict_val = dict(list_sub_dict[subjects[key_sub]])

def plot_com(dict_in, sub):
    d_val = list(dict_in.items())
    random.shuffle(d_val)
    data_val = dict(d_val)
    pd_valid = pd.concat(data_val, axis=0).reset_index(drop=True)
    plt.plot(pd_valid["no_mc_knee_angle"] - pd_valid["no_mc_kma_rel_angle"], label = sub)
    return
    

plt.figure()
plot_com(subA, "SA")
plt.figure()
plot_com(subD, "SD")
plt.figure()
plot_com(subE, "SE")
plt.figure()
plot_com(subF, "SF")
plt.figure()
plot_com(subG, "SG")
plt.figure()
plot_com(subH, "SH")


#%%

def plot_for_one_sub(dict_in):
    plt.figure()
    for k in dict_in:
        plt.plot(dict_in[k]["no_mc_shank_angle"] - dict_in[k]["no_mc_kmal_angle"])
    return


# plot_for_one_sub(fa11)
# plot_for_one_sub(fa12)
# plot_for_one_sub(fa31)
# plot_for_one_sub(fa32)
# plot_for_one_sub(fa51)
# plot_for_one_sub(fa52)

#%%

def plot_for_one_sub_s(dict_in):
    plt.figure()
    for k in dict_in:
        plt.plot(dict_in[k]["no_mc_shank_angle"])
    return

# plot_for_one_sub_s(fa11)
# plot_for_one_sub_s(fa12)
# plot_for_one_sub_s(fa31)
# plot_for_one_sub_s(fa32)
# plot_for_one_sub_s(fa51)
# plot_for_one_sub_s(fa52)

#%%

def get_mean_d(dict_in, where):
      
    arr_y = dict_in[0][where]
    for key in dict_in:
        arr_y = np.column_stack((arr_y, dict_in[key][where]))
        
    arr_y = arr_y[:,1:]
    y_mean = arr_y.mean(axis = 1)
    y_var = arr_y.var(axis =1)
    
    # plt.figure()
    # plt.plot(y_mean, label = "mean")
    # plt.plot(y_var, label = "variance")
    # plt.legend()
    return y_mean, y_var

def get_mean_diff(dict_in):
      
    arr_y = dict_in[0]["no_mc_shank_angle"] - dict_in[0]["no_mc_kmal_angle"]
    # arr_y = dict_in[0]["no_mc_knee_angle"] - dict_in[0]["no_mc_kma_rel_angle"]
    for key in dict_in:
        arr_y = np.column_stack((arr_y, dict_in[key]["no_mc_shank_angle"] - dict_in[key]["no_mc_kmal_angle"]))
        # arr_y = np.column_stack((arr_y, dict_in[key]["no_mc_knee_angle"] - dict_in[key]["no_mc_kma_rel_angle"]))
        
    arr_y = arr_y[:,1:]
    y_mean = arr_y.mean(axis = 1)
    y_var = arr_y.var(axis =1)
    
    # plt.figure()
    # plt.plot(y_mean, label = "mean")
    # plt.plot(y_var, label = "variance")
    # plt.legend()
    return y_mean, y_var
#%%

# get_mean_d(fa11,"no_mc_shank_angle")
# get_mean_d(fa12,"no_mc_shank_angle")
# get_mean_d(fa31,"no_mc_shank_angle")
# get_mean_d(fa32,"no_mc_shank_angle")
# get_mean_d(fa51,"no_mc_shank_angle")
# get_mean_d(fa52,"no_mc_shank_angle")

# get_mean_d(fa11,"no_mc_kmal_angle")
# get_mean_d(fa12,"no_mc_kmal_angle")
# get_mean_d(fa31,"no_mc_kmal_angle")
# get_mean_d(fa32,"no_mc_kmal_angle")
# get_mean_d(fa51,"no_mc_kmal_angle")
# get_mean_d(fa52,"no_mc_kmal_angle")


# get_mean_diff(fa11)
# get_mean_diff(fa12)
# get_mean_diff(fa31)
# get_mean_diff(fa32)
# get_mean_diff(fa51)
# get_mean_diff(fa52)

#%%
colors = ["b", "g", "r", "c", "m", "k"]

def plt_together(list_dict):
    f, (ax1,ax2,ax3)= plt.subplots(3, sharex=True)
    counter = 0
    for x in range(len(list_dict)):
        pr_d = list_dict[x]
        mean_shank, var_shank = get_mean_d(pr_d, "no_mc_shank_angle")
        mean_kmal, var_kmal = get_mean_d(pr_d, "no_mc_kmal_angle")
        mean_diff, var_diff = get_mean_diff(pr_d)
        ax1.plot(mean_shank, label = counter, color = colors[counter])
        ax1.plot(var_shank, "--", label = counter, color = colors[counter])
        ax2.plot(mean_kmal, label = counter, color = colors[counter])
        ax2.plot(var_kmal, "--", label = counter, color = colors[counter])
        ax3.plot(mean_diff, label = counter, color = colors[counter])
        ax3.plot(var_diff, "--", label = counter, color = colors[counter])
        counter = counter + 1
    return

l_sub = []
l_sub.append(fa11)
l_sub.append(fa12)
l_sub.append(fa31)
l_sub.append(fa32)
l_sub.append(fa51)
l_sub.append(fa52)

plt_together(l_sub)

#%%
l2 = []
l2.append(fa51)
l2.append(fa52)
l2.append(fc51)
l2.append(fc52)
l2.append(fe51)
l2.append(fh52)

plt_together(l2)



