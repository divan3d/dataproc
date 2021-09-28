# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:25:05 2021

@author: ksdiv
Cross validation
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
from sklearn.model_selection import GridSearchCV
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
data_train = dict(ds)
pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)

tests = list(test_d.items())
random.shuffle(tests)
test_val = dict(tests)
pd_test = pd.concat(test_val, axis = 0).reset_index(drop= True)

#%%

pd_train = pd_train.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
pd_test = pd_test.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])

Y_train = pd_train.pop("no_mc_shank_angle")
X_train = pd_train

Y_test = pd_test.pop("no_mc_shank_angle")
X_test = pd_test

#%%

model = xgb.XGBRegressor()
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)

#%%

grid_search = GridSearchCV(model, param_grid, n_jobs = -1)
grid_result = grid_search.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))