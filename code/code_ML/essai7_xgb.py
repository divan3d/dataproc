# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 20:54:24 2021

@author: ksdiv
check max depth and min child weight with cross validation (leave one subject out)
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

subAd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SA.pkl"
subA = op_pickle(subAd)

subDd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SD.pkl"
subD = op_pickle(subDd)

subEd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SE.pkl"
subE = op_pickle(subEd)

subFd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SF.pkl"
subF = op_pickle(subFd)

subGd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SG.pkl"
subG = op_pickle(subGd)

subHd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SH.pkl"
subH = op_pickle(subHd)

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



    pd_train = pd_train.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])
    pd_valid = pd_valid.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])
    
    Y_train = pd_train.pop("no_mc_thigh_angle")
    X_train = pd_train
    
    Y_valid = pd_valid.pop("no_mc_thigh_angle")
    X_valid = pd_valid
    
    
    
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

#%%
    # essai 1
    # max_depth = [3,5,7,9]
    max_depth = [3,5,7]
    min_child_weight = [1,3,5]
    
    # essai 2
    # max_depth = [5,6,7]
    # min_child_weight = [1,2,3,4]

#%%
    
    dico_results[key_sub] = {}
    counter = 0
    for max_depth_val in max_depth:
        for min_child_weight_val in min_child_weight:
            
            dico_results[key_sub][counter] = {}
            dico_results[key_sub][counter]["max_depth"] = max_depth_val
            dico_results[key_sub][counter]["min_child_weight"] = min_child_weight_val
            
            # model = xgb.XGBRegressor(n_estimators = 200, max_depth = max_depth_val, min_child_weight = min_child_weight_val)
            # 2 eme essai 
            model = xgb.XGBRegressor(n_estimators = 100, max_depth = max_depth_val, min_child_weight = min_child_weight_val)
            model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)
            
            Y_validtest = model.predict(X_valid)            
            validscore = mean_squared_error(Y_valid, Y_validtest)
            
            Y_traintest = model.predict(X_train)            
            trainscore = mean_squared_error(Y_train, Y_traintest)
            
            important_features = model.feature_importances_
            progress = model.evals_result()
            
            dico_results[key_sub][counter]["score validation"] = validscore
            dico_results[key_sub][counter]["score train"] = trainscore
            dico_results[key_sub][counter]["important features"] = important_features
            dico_results[key_sub][counter]["progress"] = progress
            
            print("one iter finished")
            counter = counter + 1
            
            save_obj(dico_results, "dict_max_depth_min_child_weight_thigh.pkl")

