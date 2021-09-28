# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:06:21 2021

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


f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank.pkl"
d = op_pickle(f)

test_f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank_BC.pkl"
test_d = op_pickle(test_f)

#%%
dico_results = {}


#%%

ds = list(d.items())
random.shuffle(ds)
data_train, data_valid = train_test_split(ds, test_size=0.3, random_state=42)
data_train = dict(data_train)
data_valid = dict(data_valid)
pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
pd_valid = pd.concat(data_valid, axis = 0).reset_index(drop = True)

tests = list(test_d.items())
random.shuffle(tests)
test_val = dict(tests)
pd_test = pd.concat(test_val, axis = 0).reset_index(drop= True)

#%%

pd_train = pd_train.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])
pd_valid = pd_valid.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])
pd_test = pd_test.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])

Y_train = pd_train.pop("no_mc_shank_angle")
X_train = pd_train

Y_valid = pd_valid.pop("no_mc_shank_angle")
X_valid = pd_valid

Y_test = pd_test.pop("no_mc_shank_angle")
X_test = pd_test

eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

#%%
# essai 1
# max_depth = [3,5,7,9]
# min_child_weight = [1,3,5]

# essai 2
max_depth = [5,6,7]
min_child_weight = [1,2,3,4]

#%%
counter = 0

for max_depth_val in max_depth:
    for min_child_weight_val in min_child_weight:
        
        dico_results[counter] = {}
        dico_results[counter]["max_depth"] = max_depth_val
        dico_results[counter]["min_child_weight"] = min_child_weight_val
        
        # model = xgb.XGBRegressor(n_estimators = 200, max_depth = max_depth_val, min_child_weight = min_child_weight_val)
        # 2 eme essai 
        model = xgb.XGBRegressor(n_estimators = 150, max_depth = max_depth_val, min_child_weight = min_child_weight_val)
        model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)
        
        Y_validtest = model.predict(X_valid)
        Y_pred = model.predict(X_test)
        
        validscore = mean_squared_error(Y_valid, Y_validtest)
        testscore = mean_squared_error(Y_test, Y_pred)
        important_features = model.feature_importances_
        progress = model.evals_result()
        
        dico_results[counter]["score validation"] = validscore
        dico_results[counter]["test score"] = testscore
        dico_results[counter]["important features"] = important_features
        dico_results[counter]["progress"] = progress
        
        print("one iter finished")
        counter = counter + 1
        
        save_obj(dico_results, "dict_max_depth_min_child_weight_2.pkl")

#%% plot results 
score_val = []
score_test  = []
depth_weight = []

for val in dico_results:
    score_val.append(dico_results[val]["score validation"])
    score_test.append(dico_results[val]["test score"])
    depth_weight.append([dico_results[val]["max_depth"],dico_results[val]["min_child_weight"]])
    
#%%

plt.figure()
plt.plot(score_val, "o-", label = "score on validation set")
plt.plot(score_test, "o-", label = "score on unseen test set")
plt.legend()
plt.title("effects of max depth and min child weight 2")
plt.xticks(range(12), depth_weight, rotation = 90)
    
#%% look at features

imp_feat = []
for val in dico_results:
    imp_feat.append(dico_results[val]["important features"])

list_keys = list(X_test)

#%%

plt.figure()
plt.plot(imp_feat[0])
for v in range(len(imp_feat)):
    plt.plot(imp_feat[v], "o-", label = depth_weight[v])
plt.legend()
plt.xticks(range(len(model.feature_importances_)), list_keys, rotation = 90)
plt.title("important features change when max depth and min child weight vary 2nd exp")


#%%
tot_progress =[]
for val in dico_results:
    tot_progress.append(dico_results[val]["progress"])

#%%

plt.figure()
plt.grid()
for v in range(len(tot_progress)):
    plt.plot(tot_progress[v]['validation_0']["rmse"], label='train')
    plt.plot(tot_progress[v]['validation_1']["rmse"], label='valid')
plt.legend()


