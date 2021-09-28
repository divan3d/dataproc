# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:12:20 2021

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

f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank.pkl"
d = op_pickle(f)

#%%


ds = list(d.items())
random.shuffle(ds)
data_train, data_test = train_test_split(ds, test_size=0.3, random_state=42)
data_train, data_validation = train_test_split(data_train, test_size = 0.2, random_state = 12)
data_train = dict(data_train)
data_test = dict(data_test)
data_validation = dict(data_validation)
pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
pd_test = pd.concat(data_test, axis=0).reset_index(drop=True)
pd_validation = pd.concat(data_validation, axis = 0).reset_index(drop = True)

#%% delete few columns 

# list_keys = list(pd_train)
pd_train = pd_train.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
pd_test = pd_test.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
pd_validation = pd_validation.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])

#%%

Y_train = pd_train.pop("no_mc_shank_angle")
X_train = pd_train

Y_test = pd_test.pop("no_mc_shank_angle")
X_test = pd_test

Y_validation = pd_validation.pop("no_mc_shank_angle")
X_validation = pd_validation

#%%

eval_set = [(X_train, Y_train), (X_validation, Y_validation)]


model = xgb.XGBRegressor()
model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)


Y_pred = model.predict(X_test)
#%%
score = mean_squared_error(Y_test, Y_pred)


plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()

#%%

plt.figure()
plt.plot(Y_pred, label = "Pred")
plt.plot(Y_test, label = "true")
plt.plot(X_test["no_mc_kmal_angle"], label = "input")
plt.legend()

#%%

list_used_keys = list(X_test)

#%%

# features_used = model.feature_importances_
# dict_features_used = dict((el, ) for el in features_used)
# features_used = features_used[features_used>0.01]

#%%


X_train2 = pd_train.drop(columns = "no_mc_kmal_angle")
X_test2 = pd_test.drop(columns = "no_mc_kmal_angle")
X_validation2 = pd_validation.drop(columns = "no_mc_kmal_angle")

eval_set2 = [(X_train2, Y_train), (X_validation2, Y_validation)]


model2 = xgb.XGBRegressor()
model2.fit(X_train2, Y_train, eval_set = eval_set2, verbose = True)


Y_pred2 = model2.predict(X_test2)

score2 = mean_squared_error(Y_test, Y_pred2)

#%%

plt.bar(range(len(model2.feature_importances_)), model2.feature_importances_)
plt.show()

plt.figure()
plt.plot(Y_pred2, label = "Pred")
plt.plot(Y_test, label = "true")
plt.plot(X_test["no_mc_kmal_angle"], label = "input")
plt.legend()

#%%

list2 = list(X_test2)
