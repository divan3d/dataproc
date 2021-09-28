# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:13:15 2021

@author: ksdiv
"simple" xgb model, y : segment, input all the rest
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

test_f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank_BC.pkl"
test_d = op_pickle(test_f)

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

pd_train = pd_train.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
pd_valid = pd_valid.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
pd_test = pd_test.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])

Y_train = pd_train.pop("no_mc_shank_angle")
X_train = pd_train

Y_valid = pd_valid.pop("no_mc_shank_angle")
X_valid = pd_valid

Y_test = pd_test.pop("no_mc_shank_angle")
X_test = pd_test

eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

#%%
model = xgb.XGBRegressor(n_estimators = 350)
model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)


Y_pred = model.predict(X_test)

score = mean_squared_error(Y_test, Y_pred)

list_keys = list(X_test)

plt.figure()
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.xticks(range(len(model.feature_importances_)), list_keys, rotation = 90)
plt.show()

#%%

plt.figure()
plt.plot(Y_pred, label = "prediction")
plt.plot(Y_test, label = "true")
plt.plot(X_test["no_mc_kmal_angle"], label = "kma input")
plt.legend()

#%%
import matplotlib.pyplot as plt
results = model.evals_result()
#%%
plt.figure()
plt.grid()
plt.plot(results['validation_0']["rmse"], label='train')
plt.plot(results['validation_1']["rmse"], label='test')



