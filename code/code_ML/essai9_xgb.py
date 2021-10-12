# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:24:34 2021

@author: ksdiv
regularization params
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

feat_drop = l_drop_features["remove 95%"]

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

    pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    
    Y_train = pd_train.pop("no_mc_shank_angle") 
    pd_train = pd_train.drop(columns = feat_drop)
    X_train = pd_train
    
    Y_valid = pd_valid.pop("no_mc_shank_angle") 
    pd_valid = pd_valid.drop(columns = feat_drop)
    X_valid = pd_valid
        
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

    dico_results[key_sub] = {}
    
    model = xgb.XGBRegressor(n_estimators = 200)
    model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)
    
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
    
    save_obj(dico_results, "shank_error_n_est_1.pkl")
#%% plot
colors = ["b", "g", "r", "c", "m", "k"]

count = 0
plt.figure()
for sub in dico_results:    

    plt.grid()
    plt.plot(dico_results[sub]["progress"]['validation_0']["rmse"], "o-", label='train '+ sub, color = colors[count])
    plt.plot(dico_results[sub]["progress"]['validation_1']["rmse"], "x-", label='valid '+ sub, color = colors[count])
    plt.legend()
    plt.title("shank")
    count = count + 1
    
#%%
dico_results = {}
    
#%%
# create train and validation set 
# subjects = ["SA", "SD", "SE","SF","SG","SH"]
subjects = {"SA" : 0, "SD" :1, "SE" :2,"SF" :3,"SG" :4,"SH" :5}

feat_drop = l_drop_features_knee["remove 95%"]

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

    pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    
    Y_train = pd_train.pop("no_mc_knee_angle") 
    pd_train = pd_train.drop(columns = feat_drop)
    X_train = pd_train
    
    Y_valid = pd_valid.pop("no_mc_knee_angle") 
    pd_valid = pd_valid.drop(columns = feat_drop)
    X_valid = pd_valid
        
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

    dico_results[key_sub] = {}
    
    model = xgb.XGBRegressor(n_estimators = 200)
    model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)
    
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
    
    save_obj(dico_results, "knee_error_n_est_1.pkl")
#%% plot
colors = ["b", "g", "r", "c", "m", "k"]

count = 0
plt.figure()
for sub in dico_results:    

    plt.grid()
    plt.plot(dico_results[sub]["progress"]['validation_0']["rmse"], "o-", label='train '+ sub, color = colors[count])
    plt.plot(dico_results[sub]["progress"]['validation_1']["rmse"], "x-", label='valid '+ sub, color = colors[count])
    plt.legend()
    plt.title("knee")
    count = count + 1
#%%
    # essai 1
    # max_depth = [3,5,7,9]
    # max_depth_l = [3,5]
    # max_depth = [4,5,6,7]
    
    # reg_alpha_l = [1e-3, 0.1, 1, 100]
    
    # gamma_l = [i/10.0 for i in range(0,5)]
    
    # colsample_l = [0.6,0.7,0.8,0.9]
    
    # subsample_l = [0.6,0.7,0.8,0.9]
    
    # essai 2
    # max_depth = [5,6,7]
    # min_child_weight = [1,2,3,4]

#%%
    
    # dico_results[key_sub] = {}
    # counter = 0
    # for colsample_val in colsample_l:
    #     for subsample_val in subsample_l:
            
    #         dico_results[key_sub][counter] = {}
    #         dico_results[key_sub][counter]["col_sample"] = colsample_val
    #         dico_results[key_sub][counter]["subsample"] = subsample_val
            
    #         # model = xgb.XGBRegressor(n_estimators = 200, max_depth = max_depth_val, min_child_weight = min_child_weight_val)
    #         # 2 eme essai 
    #         model = xgb.XGBRegressor(n_estimators = 200)
    #         model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 5)
            
    #         print(model.best_ntree_limit)
            
    #         Y_validtest = model.predict(X_valid)            
    #         validscore = mean_squared_error(Y_valid, Y_validtest)
          
    #         Y_traintest = model.predict(X_train)            
    #         trainscore = mean_squared_error(Y_train, Y_traintest)
            
    #         important_features = model.feature_importances_
    #         progress = model.evals_result()
            
    #         dico_results[key_sub][counter]["score validation"] = validscore
    #         dico_results[key_sub][counter]["score train"] = trainscore
    #         dico_results[key_sub][counter]["important features"] = important_features
    #         dico_results[key_sub][counter]["progress"] = progress
    #         dico_results[key_sub][counter]["best n iter"] = model.best_ntree_limit
            
    #         print("one iter finished")
    #         counter = counter + 1
            
    #         save_obj(dico_results, "thigh_error_n_est_1.pkl")

#%%

# Y_validtest = model.predict(X_valid)            
            
# Y_traintest = model.predict(X_train)   

# plt.figure()
# plt.plot(Y_train)
# plt.plot(Y_traintest)

# plt.figure()
# plt.plot(Y_valid)
# plt.plot(Y_validtest)


