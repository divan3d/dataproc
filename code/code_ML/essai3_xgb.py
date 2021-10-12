# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:13:15 2021

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
from sklearn.model_selection import GridSearchCV

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_knee_n_BC.pkl"
d = op_pickle(f)

# test_f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank_BC.pkl"
# test_d = op_pickle(test_f)

#%%

ds = list(d.items())
random.shuffle(ds)
# data_train, data_valid = train_test_split(ds, test_size=0.3, random_state=42)
data_train = dict(ds)
# data_valid = dict(data_valid)
pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
# pd_valid = pd.concat(data_valid, axis = 0).reset_index(drop = True)

# tests = list(test_d.items())
# random.shuffle(tests)
# test_val = dict(tests)
# pd_test = pd.concat(test_val, axis = 0).reset_index(drop= True)

#%%

pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
# pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
# pd_test = pd_test.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])

Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
X_train = pd_train


#%%
model = xgb.XGBRegressor(n_estimators = 150)
# model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 5)


# n_estimators_val = [30,50,100,200]
# param_grid = dict(n_estimators = n_estimators_val)

max_depth_val = [3,5]
min_child_weight_val = [1,3,5]
param_grid = dict(max_depth = max_depth_val, min_child_weight = min_child_weight_val)



grid_search = GridSearchCV(model, param_grid, n_jobs = -1, scoring = "neg_mean_squared_error")
grid_result = grid_search.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# Y_pred = model.predict(X_test)

# score = mean_squared_error(Y_test, Y_pred)

# list_keys = list(X_train)

# plt.figure()
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plt.xticks(range(len(model.feature_importances_)), list_keys, rotation = 90)
# plt.show()

#%%
save_obj(grid_result, "shank_compliance_gscv_results_2.pkl")


#%%



#%%
# list_keys = list(X_train)
# a = grid_search.best_estimator_.feature_importances_
# b = np.column_stack((a,list_keys))
# b[b[:, 0].argsort()]

# order_feature = b[:,1]
# imp_feature = order_feature[-15:]
# not_imp_feature = order_feature[:-15]
# l_not_imp_feature = list(not_imp_feature)

# plt.figure()
# plt.bar(range(len(list_keys)), a)
# plt.xticks(range(len(list_keys)), list_keys, rotation = 90)
# plt.show()

# save_obj(l_not_imp_feature, "list_features_to_dump.pkl")

#%%

# plt.figure()
# plt.plot(Y_pred, label = "prediction")
# plt.plot(Y_test, label = "true")
# plt.plot(X_test["no_mc_kmal_angle"], label = "kma input")
# plt.legend()

#%%
# import matplotlib.pyplot as plt
# results = model.evals_result()
# #%%
# plt.figure()
# plt.grid()
# plt.plot(results['validation_0']["rmse"], label='train')
# plt.plot(results['validation_1']["rmse"], label='valid')



