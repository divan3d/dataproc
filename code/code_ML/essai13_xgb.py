# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:12:01 2021

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

subAd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SA.pkl"
subA = op_pickle(subAd)

subDd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SD.pkl"
subD = op_pickle(subDd)

subEd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SE.pkl"
subE = op_pickle(subEd)

subFd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SF.pkl"
subF = op_pickle(subFd)

subGd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SG.pkl"
subG = op_pickle(subGd)

subHd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SH.pkl"
subH = op_pickle(subHd)

l_drop_features_d = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_shank.pkl"
l_drop_features = op_pickle(l_drop_features_d)

l_drop_features_dknee = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_knee.pkl"
l_drop_features_knee = op_pickle(l_drop_features_dknee)

l_drop_d_t =  r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_thigh.pkl"
drop_thigh = op_pickle(l_drop_d_t)
#%%
dico_results = {}

#%%

list_sub_dict = []
list_sub_dict.append(subA)
list_sub_dict.append(subD)
list_sub_dict.append(subE)
list_sub_dict.append(subF)
list_sub_dict.append(subG)
list_sub_dict.append(subH)

#%%
# create train and validation set 
# subjects = ["SA", "SD", "SE","SF","SG","SH"]
subjects = {"SA" : 0, "SD" :1, "SE" :2,"SF" :3,"SG" :4,"SH" :5}

feat_drop = drop_thigh["remove 95%"]

# choose one 


for key_sub in subjects :
    print("validation subject:")
    print(key_sub)
    sub_train = subjects.copy()
    del sub_train[key_sub]
    
    d_train = dict()
    for each_sub in sub_train:
        d_train.update(list_sub_dict[subjects[each_sub]])
        
    dict_val = dict(list_sub_dict[subjects[key_sub]])

    ds = list(d_train.items())
    random.shuffle(ds)
    data_train = dict(ds)
    pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
    
    d_val = list(dict_val.items())
    random.shuffle(d_val)
    data_val = dict(d_val)
    pd_valid = pd.concat(data_val, axis=0).reset_index(drop=True)

    pd_train = pd_train.drop(columns = ["t",  "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    pd_valid = pd_valid.drop(columns = ["t",  "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    
    Y_train = pd_train.pop("no_mc_thigh_angle") 
    pd_train = pd_train.drop(columns = feat_drop)
    X_train = pd_train
    
    Y_valid = pd_valid.pop("no_mc_thigh_angle") 
    pd_valid = pd_valid.drop(columns = feat_drop)
    X_valid = pd_valid
        
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

    dico_results[key_sub] = {}
    
    # model = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 1000, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_lambda = 100, subsample = 0.9, colsample_bytree = 0.7)
    # model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 50)
    
    # model = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 1000, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_alpha = 1, subsample = 0.9, colsample_bytree = 0.9, reg_lambda = 0.1)
    # model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 50)
    
    model = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 1000, max_depth = 8, min_child_weight = 1, min_split_loss = 0.1, reg_alpha = 0.1, subsample = 0.8, colsample_bytree = 0.9)
    model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 50)
    
    print(model.best_ntree_limit)
    
    Y_validtest = model.predict(X_valid)            
    validscore = mean_squared_error(Y_valid, Y_validtest)
  
    Y_traintest = model.predict(X_train)            
    trainscore = mean_squared_error(Y_train, Y_traintest)
    
    important_features = model.feature_importances_
    progress = model.evals_result()
    
    dico_results[key_sub]["score validation"] = validscore
    dico_results[key_sub]["score train"] = trainscore
    dico_results[key_sub]["important features"] = important_features
    dico_results[key_sub]["progress"] = progress
    dico_results[key_sub]["best n iter"] = model.best_ntree_limit
    
    print("one iter finished")
    
    save_obj(dico_results, "thigh_error_n_est_opt_1.pkl")
    
#%% plot
colors = ["b", "g", "r", "c", "m", "k"]

count = 0
plt.figure()
for sub in dico_results:    

    plt.grid()
    # plt.plot(dico_results[sub]["progress"]['validation_0']["rmse"], "o-", label='train '+ sub, color = colors[count])
    plt.plot(dico_results[sub]["progress"]['validation_1']["rmse"], "x-", label='valid '+ sub, color = colors[count])
    plt.legend()
    plt.title("thigh")
    count = count + 1