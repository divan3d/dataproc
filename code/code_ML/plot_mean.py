# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:13:49 2021

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
import scipy.signal
import xgboost as xgb 
import math

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
subBd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SB.pkl"
subB = op_pickle(subBd)

subCd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SC.pkl"
subC = op_pickle(subCd)

l_drop_features_d = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_shank.pkl"
l_drop_features = op_pickle(l_drop_features_d)

l_drop_features_k = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_knee.pkl"
l_drop_featuresk = op_pickle(l_drop_features_k)

l_drop_features_t = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_thigh.pkl"
l_drop_featurest = op_pickle(l_drop_features_t)

#%%
def indices_to_cut_R(d1):
    """
    finds indices to cut data such that left with ~ heel strike to toe off for 
    right leg -- corresponds approx. to heel strike right leg to heel strike left leg 

    Parameters
    ----------
    d1 : pd dataframe : dynamic data of 1 subject

    Returns
    -------
    peak_R_leg : array : contains indices of Right leg heel strike 

    """
    _, properties = scipy.signal.find_peaks(d1.values, height=16, plateau_size = 5)
    peak_R_leg = properties["right_edges"]
    return peak_R_leg

def indices_to_cut_R_early(d1):
    """
    finds indices to cut data such that left with ~ heel strike to toe off for 
    right leg -- corresponds approx. to heel strike right leg to heel strike left leg 

    Parameters
    ----------
    d1 : pd dataframe : dynamic data of 1 subject

    Returns
    -------
    peak_R_leg : array : contains indices of Right leg heel strike 

    """
    _, properties = scipy.signal.find_peaks(d1.values, height=16, plateau_size = 5)
    peak_R_leg = properties["right_edges"]
    scd_cut = peak_R_leg + 156
    return peak_R_leg, scd_cut


#%%

def get_mean_thigh(test_data, y_data, y_pred):
    new_df = {}
    new_df["no_mc_kmau_angle"] = test_data["no_mc_kmau_angle"]
    new_df["dyn_eq_phase"] = test_data["dyn_eq_phase"]
    new_df["y_pred"] = y_pred
    new_df["y_true"] = y_data
    
    new_df = pd.DataFrame.from_dict(new_df)
    
    ess = indices_to_cut_R(new_df["dyn_eq_phase"])
    ess2 = ess + 1
    
    dict_cut_y_true = {}
    dict_cut_y_pred = {}
    dict_cut_no_mc_kmal_angle = {}
    for cidx in range(len(ess2)-1):
            dict_cut_y_true[cidx] = new_df["y_true"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
            dict_cut_y_pred[cidx] = new_df["y_pred"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
            dict_cut_no_mc_kmal_angle[cidx] = new_df["no_mc_kmau_angle"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
    
    # plt.figure()
    # plt.plot(new_df["dyn_eq_phase"])
    # plt.plot(ess, new_df["dyn_eq_phase"][ess], "o")

    
    arr_y_pred = dict_cut_y_pred[0]
    for key in dict_cut_y_pred:
        arr_y_pred = np.column_stack((arr_y_pred, dict_cut_y_pred[key]))
        
    arr_y_pred = arr_y_pred[:,1:]
    y_pred_mean = arr_y_pred.mean(axis = 1)
    y_pred_var = np.var(arr_y_pred, axis = 1)
    y_pred_stdrd_dev = np.sqrt(y_pred_var)
    
    arr_y_true = dict_cut_y_true[0]
    for key in dict_cut_y_true:
        arr_y_true = np.column_stack((arr_y_true, dict_cut_y_true[key]))
        
    arr_y_true = arr_y_true[:,1:]
    y_true_mean = arr_y_true.mean(axis = 1)
    y_true_var = np.var(arr_y_true, axis = 1)
    y_true_stdrd_dev = np.sqrt(y_true_var)
        

    arr_x = dict_cut_no_mc_kmal_angle[0]
    for key in dict_cut_no_mc_kmal_angle:
        arr_x = np.column_stack((arr_x, dict_cut_no_mc_kmal_angle[key]))
        
    arr_x = arr_x[:,1:]
    x_mean = arr_x.mean(axis = 1)
    x_var = np.var(arr_x, axis = 1)
    x_stdrd_dev = np.sqrt(x_var)
    
    len_plot = 170
    
    plt.figure()
    plt.plot(np.linspace(0,100,len_plot),y_pred_mean, label = "Prediction")
    # plt.plot(np.linspace(0,100,len_plot),y_pred_mean + y_pred_stdrd_dev, color = "lightsteelblue")
    # plt.plot(np.linspace(0,100,len_plot),y_pred_mean - y_pred_stdrd_dev, color = "lightsteelblue")
    # plt.fill_between(np.linspace(0,100,len_plot), y_pred_mean - y_pred_stdrd_dev,y_pred_mean + y_pred_stdrd_dev, color = "lightsteelblue", label = "Standard deviation")
    plt.plot(np.linspace(0,100,len_plot),y_true_mean, label = "True")
    # plt.plot(np.linspace(0,100,len_plot),y_true_mean + y_true_stdrd_dev, color = "wheat")
    # plt.plot(np.linspace(0,100,len_plot),y_true_mean - y_true_stdrd_dev, color = "wheat")
    # plt.fill_between(np.linspace(0,100,len_plot), y_true_mean - y_true_stdrd_dev,y_true_mean + y_true_stdrd_dev, color = "wheat")
    plt.plot(np.linspace(0,100,len_plot),x_mean, label = "Input")
    # plt.plot(np.linspace(0,100,len_plot),x_mean + x_stdrd_dev, color = "palegreen")
    # plt.plot(np.linspace(0,100,len_plot),x_mean - x_stdrd_dev, color = "palegreen")
    # plt.fill_between(np.linspace(0,100,len_plot), x_mean - x_stdrd_dev,x_mean + x_stdrd_dev, color = "palegreen", label = "Standard deviation")
    plt.xlim([0,100])
    plt.ylim([-45,45])
    plt.title("Mean Gait Cycle Thigh")
    plt.legend()
    plt.xlabel("Percentage gait cycle [%]")
    plt.ylabel("angle [deg]")
    return

def get_mean_knee(test_data, y_data, y_pred):
    new_df = {}
    new_df["no_mc_kma_rel_angle"] = test_data["no_mc_kma_rel_angle"]
    new_df["dyn_eq_phase"] = test_data["dyn_eq_phase"]
    new_df["y_pred"] = y_pred
    new_df["y_true"] = y_data
    
    new_df = pd.DataFrame.from_dict(new_df)
    
    ess = indices_to_cut_R(new_df["dyn_eq_phase"])
    ess2 = ess + 1
    
    dict_cut_y_true = {}
    dict_cut_y_pred = {}
    dict_cut_no_mc_kmal_angle = {}
    for cidx in range(len(ess2)-1):
            dict_cut_y_true[cidx] = new_df["y_true"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
            dict_cut_y_pred[cidx] = new_df["y_pred"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
            dict_cut_no_mc_kmal_angle[cidx] = new_df["no_mc_kma_rel_angle"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
    
    # plt.figure()
    # plt.plot(new_df["dyn_eq_phase"])
    # plt.plot(ess, new_df["dyn_eq_phase"][ess], "o")

    
    arr_y_pred = dict_cut_y_pred[0]
    for key in dict_cut_y_pred:
        arr_y_pred = np.column_stack((arr_y_pred, dict_cut_y_pred[key]))
        
    arr_y_pred = arr_y_pred[:,1:]
    y_pred_mean = arr_y_pred.mean(axis = 1)
    y_pred_var = np.var(arr_y_pred, axis = 1)
    y_pred_stdrd_dev = np.sqrt(y_pred_var)
    
    arr_y_true = dict_cut_y_true[0]
    for key in dict_cut_y_true:
        arr_y_true = np.column_stack((arr_y_true, dict_cut_y_true[key]))
        
    arr_y_true = arr_y_true[:,1:]
    y_true_mean = arr_y_true.mean(axis = 1)
    y_true_var = np.var(arr_y_true, axis = 1)
    y_true_stdrd_dev = np.sqrt(y_true_var)
        

    arr_x = dict_cut_no_mc_kmal_angle[0]
    for key in dict_cut_no_mc_kmal_angle:
        arr_x = np.column_stack((arr_x, dict_cut_no_mc_kmal_angle[key]))
        
    arr_x = arr_x[:,1:]
    x_mean = arr_x.mean(axis = 1)
    x_var = np.var(arr_x, axis = 1)
    x_stdrd_dev = np.sqrt(x_var)
    
    len_plot = 170
    
    plt.figure()
    plt.plot(np.linspace(0,100,len_plot),y_pred_mean, label = "Prediction")
    plt.plot(np.linspace(0,100,len_plot),y_pred_mean + y_pred_stdrd_dev, color = "lightsteelblue")
    plt.plot(np.linspace(0,100,len_plot),y_pred_mean - y_pred_stdrd_dev, color = "lightsteelblue")
    plt.fill_between(np.linspace(0,100,len_plot), y_pred_mean - y_pred_stdrd_dev,y_pred_mean + y_pred_stdrd_dev, color = "lightsteelblue", label = "Standard deviation")
    plt.plot(np.linspace(0,100,len_plot),y_true_mean, label = "True")
    # plt.plot(np.linspace(0,100,len_plot),y_true_mean + y_true_stdrd_dev, color = "wheat")
    # plt.plot(np.linspace(0,100,len_plot),y_true_mean - y_true_stdrd_dev, color = "wheat")
    # plt.fill_between(np.linspace(0,100,len_plot), y_true_mean - y_true_stdrd_dev,y_true_mean + y_true_stdrd_dev, color = "wheat")
    plt.plot(np.linspace(0,100,len_plot),x_mean, label = "Input")
    plt.plot(np.linspace(0,100,len_plot),x_mean + x_stdrd_dev, color = "palegreen")
    plt.plot(np.linspace(0,100,len_plot),x_mean - x_stdrd_dev, color = "palegreen")
    plt.fill_between(np.linspace(0,100,len_plot), x_mean - x_stdrd_dev,x_mean + x_stdrd_dev, color = "palegreen", label = "Standard deviation")
    plt.xlim([0,100])
    plt.ylim([-10,80])
    plt.title("Mean Gait Cycle Knee")
    plt.legend()
    plt.xlabel("Percentage gait cycle [%]")
    plt.ylabel("angle [deg]")
    return


def get_mean_shank(test_data, y_data, y_pred, dyn_phase):
    new_df = {}
    new_df["no_mc_kmal_angle"] = test_data["no_mc_kmal_angle"]
    new_df["dyn_eq_phase"] = dyn_phase
    new_df["y_pred"] = y_pred
    new_df["y_true"] = y_data
    
    new_df = pd.DataFrame.from_dict(new_df)
    
    # ess = indices_to_cut_R(new_df["dyn_eq_phase"])
    ess, snd_cut = indices_to_cut_R_early(new_df["dyn_eq_phase"])
    ess2 = ess + 1
    
    dict_cut_y_true = {}
    dict_cut_y_pred = {}
    dict_cut_no_mc_kmal_angle = {}
    
    # for cidx in range(len(ess2)-1):
    #         dict_cut_y_true[cidx] = new_df["y_true"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
    #         dict_cut_y_pred[cidx] = new_df["y_pred"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
    #         dict_cut_no_mc_kmal_angle[cidx] = new_df["no_mc_kmal_angle"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
    
    for cidx in range(len(ess2)):
            dict_cut_y_true[cidx] = new_df["y_true"].iloc[ess2[cidx]: snd_cut[cidx]].to_numpy()
            dict_cut_y_pred[cidx] = new_df["y_pred"].iloc[ess2[cidx]: snd_cut[cidx]].to_numpy()
            dict_cut_no_mc_kmal_angle[cidx] = new_df["no_mc_kmal_angle"].iloc[ess2[cidx]: snd_cut[cidx]].to_numpy()
    
    # plt.figure()
    # plt.plot(new_df["dyn_eq_phase"])
    # plt.plot(ess, new_df["dyn_eq_phase"][ess], "o")

    
    arr_y_pred = dict_cut_y_pred[0]
    for key in dict_cut_y_pred:
        arr_y_pred = np.column_stack((arr_y_pred, dict_cut_y_pred[key]))
        
    arr_y_pred = arr_y_pred[:,1:]
    y_pred_mean = arr_y_pred.mean(axis = 1)
    y_pred_var = np.var(arr_y_pred, axis = 1)
    y_pred_stdrd_dev = np.sqrt(y_pred_var)
    
    arr_y_true = dict_cut_y_true[0]
    for key in dict_cut_y_true:
        arr_y_true = np.column_stack((arr_y_true, dict_cut_y_true[key]))
        
    arr_y_true = arr_y_true[:,1:]
    y_true_mean = arr_y_true.mean(axis = 1)
    y_true_var = np.var(arr_y_true, axis = 1)
    y_true_stdrd_dev = np.sqrt(y_true_var)
        
    arr_x = dict_cut_no_mc_kmal_angle[0]
    for key in dict_cut_no_mc_kmal_angle:
        arr_x = np.column_stack((arr_x, dict_cut_no_mc_kmal_angle[key]))
        
    arr_x = arr_x[:,1:]
    x_mean = arr_x.mean(axis = 1)
    x_var = np.var(arr_x, axis = 1)
    x_stdrd_dev = np.sqrt(x_var)
    
    len_plot = 155
    # len_plot = 170
    
    plt.figure()
    plt.plot(np.linspace(0,100,len_plot),y_pred_mean, label = "Prediction")
    # plt.plot(np.linspace(0,100,len_plot),y_pred_mean + y_pred_stdrd_dev, color = "lightsteelblue")
    # plt.plot(np.linspace(0,100,len_plot),y_pred_mean - y_pred_stdrd_dev, color = "lightsteelblue")
    plt.fill_between(np.linspace(0,100,len_plot), y_pred_mean - y_pred_stdrd_dev,y_pred_mean + y_pred_stdrd_dev,color = "white", edgecolor='lightsteelblue', hatch='*', label = "Standard deviation")
    
    # plt.fill_between(np.linspace(0,100,len_plot), y_pred_mean - y_pred_stdrd_dev,y_pred_mean + y_pred_stdrd_dev, color = "lightsteelblue", label = "Standard deviation")
    plt.plot(np.linspace(0,100,len_plot),y_true_mean, label = "True")
    # plt.plot(np.linspace(0,100,len_plot),y_true_mean + y_true_stdrd_dev, color = "wheat")
    # plt.plot(np.linspace(0,100,len_plot),y_true_mean - y_true_stdrd_dev, color = "wheat")
    # plt.fill_between(np.linspace(0,100,len_plot), y_true_mean - y_true_stdrd_dev,y_true_mean + y_true_stdrd_dev, color = "wheat")
    plt.plot(np.linspace(0,100,len_plot),x_mean, label = "Input")
    # plt.plot(np.linspace(0,100,len_plot),x_mean + x_stdrd_dev, color = "palegreen")
    # plt.plot(np.linspace(0,100,len_plot),x_mean - x_stdrd_dev, color = "palegreen")
    plt.fill_between(np.linspace(0,100,len_plot), x_mean - x_stdrd_dev,x_mean + x_stdrd_dev, edgecolor='palegreen', hatch='\\', label = "Standard deviation")
    # plt.fill_between(np.linspace(0,100,len_plot), x_mean - x_stdrd_dev,x_mean + x_stdrd_dev,color = "palegreen", label = "Standard deviation")
    plt.xlim([0,100])
    plt.ylim([-60,40])
    plt.title("Mean Gait Cycle Shank")
    plt.legend()
    plt.xlabel("Percentage gait cycle [%]")
    plt.ylabel("angle [deg]")
    return


#%%
# get_mean_d(pd_C_thigh, Y_C_thigh, thigh_pred_C)
# get_mean_d(pd_B_thigh, Y_B_thigh, thigh_pred_B)

#%% load models

mod_shank = xgb.Booster()
mod_shank.load_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/shank_1.model")

mod_thigh = xgb.Booster()
mod_thigh.load_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/thigh_1.model")

mod_knee = xgb.Booster()
mod_knee.load_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/knee_1.model")

#%% load data

d_testC = list(subC.items())
random.shuffle(d_testC)
test_C = dict(d_testC)
pd_testC = pd.concat(test_C, axis = 0).reset_index(drop= True)

d_testB = list(subB.items())
random.shuffle(d_testB)
test_B = dict(d_testB)
pd_testB = pd.concat(test_B, axis = 0).reset_index(drop= True)

#%% load data thigh

pd_C_thigh = pd_testC.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_C_thigh = pd_C_thigh.pop("no_mc_thigh_angle") 
pd_C_thigh = pd_C_thigh.drop(columns = l_drop_featurest["remove 95%"])

pd_B_thigh = pd_testB.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_B_thigh = pd_B_thigh.pop("no_mc_thigh_angle") 
pd_B_thigh = pd_B_thigh.drop(columns = l_drop_featurest["remove 95%"])

#%% load data shank

pd_C_shank = pd_testC.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_C_shank = pd_C_shank.pop("no_mc_shank_angle") 
C_shank_dyn_eq = pd_C_shank.pop("dyn_eq_phase")
shank_feature_drop = l_drop_features["remove 95%"]
index = np.argwhere(shank_feature_drop== "dyn_eq_phase")
shank_feature_drop = np.delete(shank_feature_drop, index[0][0])
pd_C_shank = pd_C_shank.drop(columns = shank_feature_drop)

pd_B_shank = pd_testB.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_B_shank = pd_B_shank.pop("no_mc_shank_angle") 
B_shank_dyn_eq = pd_B_shank.pop("dyn_eq_phase")
pd_B_shank = pd_B_shank.drop(columns = shank_feature_drop)

#%% load data knee

pd_C_knee = pd_testC.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_C_knee = pd_C_knee.pop("no_mc_knee_angle") 
pd_C_knee = pd_C_knee.drop(columns = l_drop_featuresk["remove 95%"])

pd_B_knee = pd_testB.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_B_knee = pd_B_knee.pop("no_mc_knee_angle") 
pd_B_knee = pd_B_knee.drop(columns = l_drop_featuresk["remove 95%"])

#%% D matrix
D_C_thigh = xgb.DMatrix(pd_C_thigh, label = Y_C_thigh)
D_B_thigh = xgb.DMatrix(pd_B_thigh, label = Y_B_thigh)

D_C_shank = xgb.DMatrix(pd_C_shank, label = Y_C_shank)
D_B_shank = xgb.DMatrix(pd_B_shank, label = Y_B_shank)

D_C_knee = xgb.DMatrix(pd_C_knee, label = Y_C_knee)
D_B_knee = xgb.DMatrix(pd_B_knee, label = Y_B_knee)


#%% get predictions 

thigh_pred_C = mod_thigh.predict(D_C_thigh)
thigh_pred_B = mod_thigh.predict(D_B_thigh)

shank_pred_C = mod_shank.predict(D_C_shank)
shank_pred_B = mod_shank.predict(D_B_shank)

knee_pred_C = mod_knee.predict(D_C_knee)
knee_pred_B = mod_knee.predict(D_B_knee)

#%% 
get_mean_thigh(pd_C_thigh,Y_C_thigh, thigh_pred_C)
get_mean_thigh(pd_B_thigh,Y_B_thigh, thigh_pred_B)
get_mean_knee(pd_C_knee,Y_C_knee, knee_pred_C)
get_mean_shank(pd_C_shank,Y_C_shank, shank_pred_C,C_shank_dyn_eq)
get_mean_shank(pd_B_shank,Y_B_shank, shank_pred_B,B_shank_dyn_eq)
