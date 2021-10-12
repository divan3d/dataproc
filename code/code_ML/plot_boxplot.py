# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:17:08 2021

@author: ksdiv
function plots boxplot
"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy import random
from sklearn.metrics import mean_squared_error
import scipy.signal
import xgboost as xgb 
import math
import statistics 

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

#%% cut function
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

def indices_to_cut_R_2(d1):
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
    _, properties = scipy.signal.find_peaks(d1.values, height=1, plateau_size = 5)
    peak_R_leg = properties["right_edges"]
    return peak_R_leg



#%% veut faire boxplot de RMSE - du coup nous faut vecteurs de RMSE par truc qu'on veut plotter
# verifier que coupe au bon endroit 

def get_rmse_thigh(test_data, y_data, y_pred):
    rmse_og = []
    rmse_prediction = []
    
    # cut 
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
    
    # rmse 
    
    for k in dict_cut_y_pred:
        temp = mean_squared_error(dict_cut_y_true[k], dict_cut_y_pred[k])
        rmse_prediction.append(math.sqrt(temp))
        
    for k in dict_cut_no_mc_kmal_angle:
        temp = mean_squared_error(dict_cut_y_true[k], dict_cut_no_mc_kmal_angle[k])
        rmse_og.append(math.sqrt(temp))
        
    
    return rmse_og, rmse_prediction

def get_rmse_knee(test_data, y_data, y_pred):
    rmse_og = []
    rmse_prediction = []
    
    # cut 
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
    
    # rmse 
    
    for k in dict_cut_y_pred:
        temp = mean_squared_error(dict_cut_y_true[k], dict_cut_y_pred[k])
        rmse_prediction.append(math.sqrt(temp))
        
    for k in dict_cut_no_mc_kmal_angle:
        temp = mean_squared_error(dict_cut_y_true[k], dict_cut_no_mc_kmal_angle[k])
        rmse_og.append(math.sqrt(temp))
        
    
    return rmse_og, rmse_prediction

def get_rmse_shank(test_data, y_data, y_pred, dyn_phase):
    rmse_og = []
    rmse_prediction = []
    
    # cut 
    new_df = {}
    new_df["no_mc_kmal_angle"] = test_data["no_mc_kmal_angle"]
    new_df["dyn_eq_phase"] = dyn_phase
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
            dict_cut_no_mc_kmal_angle[cidx] = new_df["no_mc_kmal_angle"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
    
    # rmse 
    
    for k in dict_cut_y_pred:
        temp = mean_squared_error(dict_cut_y_true[k], dict_cut_y_pred[k])
        rmse_prediction.append(math.sqrt(temp))
        
    for k in dict_cut_no_mc_kmal_angle:
        temp = mean_squared_error(dict_cut_y_true[k], dict_cut_no_mc_kmal_angle[k])
        rmse_og.append(math.sqrt(temp))
        
    
    return rmse_og, rmse_prediction

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

#%%
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

#%% get rmse

rmse_og_thigh_C, rmse_pred_thigh_C = get_rmse_thigh(pd_C_thigh, Y_C_thigh, thigh_pred_C)
rmse_og_thigh_B, rmse_pred_thigh_B = get_rmse_thigh(pd_B_thigh, Y_B_thigh, thigh_pred_B)

rmse_og_shank_C, rmse_pred_shank_C = get_rmse_shank(pd_C_shank, Y_C_shank, shank_pred_C,C_shank_dyn_eq)
rmse_og_shank_B, rmse_pred_shank_B = get_rmse_shank(pd_B_shank, Y_B_shank, shank_pred_B,B_shank_dyn_eq)

rmse_og_knee_C, rmse_pred_knee_C = get_rmse_knee(pd_C_knee, Y_C_knee, knee_pred_C)
rmse_og_knee_B, rmse_pred_knee_B = get_rmse_knee(pd_B_knee, Y_B_knee, knee_pred_B)


#%% boxplot 


# plt.figure()
# plt.plot(pd_B_shank["R_leg"])
# plt.plot(pd_B_shank["no_mc_kmal_angle"])

# plt.figure()
# plt.plot(pd_B_thigh["dyn_eq_phase"])
# plt.plot(pd_B_thigh["no_mc_kmau_angle"])


fig, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True)
bplot1 = ax1.boxplot((rmse_og_thigh_B, rmse_pred_thigh_B, rmse_og_thigh_C, rmse_pred_thigh_C), patch_artist=True, showmeans = True)
bplot2 = ax2.boxplot((rmse_og_shank_B, rmse_pred_shank_B, rmse_og_shank_C, rmse_pred_shank_C), patch_artist=True, showmeans = True)
bplot3 = ax3.boxplot((rmse_og_knee_B, rmse_pred_knee_B, rmse_og_knee_C, rmse_pred_knee_C), patch_artist=True, showmeans = True)
                    # labels = ("Sub B original","Sub B prediction", "Sub C original", "Sub C prediction", 
                    #           "_nolegend_","_nolegend_","_nolegend_","_nolegend_",
                    #           "_nolegend_","_nolegend_","_nolegend_","_nolegend_"))


colors = ["lightgreen", "mediumseagreen","lightskyblue", "royalblue"]
          # "lightgreen", "mediumseagreen","lightskyblue", "royalblue","lightgreen", "mediumseagreen","lightskyblue", "royalblue"]

for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    
    for patch, color in zip(bplot['medians'], colors):
            patch.set_color("black")
            
    for patch, color in zip(bplot['means'], colors):
            patch.set_markerfacecolor("black")
            patch.set_markeredgecolor("black")
       

ax1.set_xlabel("Thigh")
ax2.set_xlabel("Shank")
ax3.set_xlabel("Knee")
      
ax1.tick_params(axis='x', which=u'both',length=0)
ax1.set_xticks([])
ax2.tick_params(axis='x', which=u'both',length=0)
ax2.set_xticks([])
ax3.tick_params(axis='x', which=u'both',length=0)
ax3.set_xticks([])


ax1.set_ylabel("RMSE")

fig.suptitle("RMSE for test subjects")
ax1.legend([bplot["boxes"][0], bplot["boxes"][1],bplot["boxes"][2], bplot["boxes"][3], bplot['medians'][0], bplot['means'][0]], 
           ["Sub B measured orthosis angle","Sub B estimated segment angle", "Sub C measured orthosis angle", "Sub C estimated segment angle", "median", "mean"], loc='upper left')
ax2.legend([bplot["boxes"][0], bplot["boxes"][1],bplot["boxes"][2], bplot["boxes"][3], bplot['medians'][0], bplot['means'][0]], 
           ["Sub B measured orthosis angle","Sub B estimated segment angle", "Sub C measured orthosis angle", "Sub C estimated segment angle", "median", "mean"], loc='upper left')
ax3.legend([bplot["boxes"][0], bplot["boxes"][1],bplot["boxes"][2], bplot["boxes"][3], bplot['medians'][0], bplot['means'][0]], 
           ["Sub B measured orthosis angle","Sub B estimated segment angle", "Sub C measured orthosis angle", "Sub C estimated segment angle", "median", "mean"], loc='upper left')





