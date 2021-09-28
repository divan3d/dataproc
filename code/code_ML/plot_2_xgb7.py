# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:56:39 2021

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

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_thigh_n_BC.pkl"
d = op_pickle(f)

test_f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_thigh_only_BC.pkl"
test_d = op_pickle(test_f)

# test_f = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SC.pkl"
# test_d = op_pickle(test_f)

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

pd_train = pd_train.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])
pd_test = pd_test.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])

Y_train = pd_train.pop("no_mc_thigh_angle")
X_train = pd_train


Y_test = pd_test.pop("no_mc_thigh_angle")
X_test = pd_test

# eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

#%%
model = xgb.XGBRegressor(n_estimators = 50, max_depth = 5, min_child_weight = 3)
model.fit(X_train, Y_train, verbose = True)

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
plt.plot(X_test["no_mc_kmau_angle"], label = "kma input")
plt.legend()

#%%
# import matplotlib.pyplot as plt
# results = model.evals_result()
# #%%
# plt.figure()
# plt.grid()
# plt.plot(results['validation_0']["rmse"], label='train')

#%%

# from scipy import signal

# b,a = signal.butter(8, 0.2)
# y_but = signal.filtfilt(b,a, Y_pred)

# plt.figure()
# plt.plot(y_but, label = "prediction")
# plt.plot(Y_test, label = "true")
# plt.plot(X_test["no_mc_kmal_angle"], label = "kma input")
# plt.legend()

# y_C_filt = signal.filtfilt(b,a, predC)
# plt.figure()
# plt.plot(y_C_filt, label = "prediction")
# plt.plot(Y_test, label = "true")
# plt.plot(X_test["no_mc_kmal_angle"], label = "kma input")
# plt.legend()
# plt.title("train without B and C, test on sub C")

#%%

# new_df = {}
# new_df["no_mc_kmal_angle"] = X_test["no_mc_kmal_angle"]
# new_df["dyn_eq_phase"] = X_test["dyn_eq_phase"]
# new_df["y_pred_filtered"] = y_but
# new_df["y_pred"] = Y_pred
# new_df["y_true"] = Y_test

# new_df = pd.DataFrame.from_dict(new_df)

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

# ess = indices_to_cut_R(new_df["dyn_eq_phase"])
# ess2 = ess + 1

# dict_cut_y_true = {}
# dict_cut_y_pred = {}
# dict_cut_y_pred_filtered = {}
# dict_cut_no_mc_kmal_angle = {}
# for cidx in range(len(ess2)-1):
#         dict_cut_y_true[cidx] = new_df["y_true"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
#         dict_cut_y_pred[cidx] = new_df["y_pred"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()
#         dict_cut_y_pred_filtered[cidx] = new_df["y_pred_filtered"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy() 
#         dict_cut_no_mc_kmal_angle[cidx] = new_df["no_mc_kmal_angle"].iloc[ess2[cidx]: ess2[cidx + 1]].to_numpy()

# plt.figure()
# plt.plot(new_df["dyn_eq_phase"])
# plt.plot(ess, new_df["dyn_eq_phase"][ess], "o")

#%%

# arr_y_pred = dict_cut_y_pred[0]
# for key in dict_cut_y_pred:
#     arr_y_pred = np.column_stack((arr_y_pred, dict_cut_y_pred[key]))
    
# arr_y_pred = arr_y_pred[:,1:]
# y_pred_mean = arr_y_pred.mean(axis = 1)

# arr_y_true = dict_cut_y_true[0]
# for key in dict_cut_y_true:
#     arr_y_true = np.column_stack((arr_y_true, dict_cut_y_true[key]))
    
# arr_y_true = arr_y_true[:,1:]
# y_true_mean = arr_y_true.mean(axis = 1)

# arr_y_pred_filtered = dict_cut_y_pred_filtered[0]
# for key in dict_cut_y_pred_filtered:
#     arr_y_pred_filtered = np.column_stack((arr_y_pred_filtered, dict_cut_y_pred_filtered[key]))
    
# arr_y_pred_filtered = arr_y_pred_filtered[:,1:]
# y_pred_filtered_mean = arr_y_pred_filtered.mean(axis = 1)

# arr_x = dict_cut_no_mc_kmal_angle[0]
# for key in dict_cut_no_mc_kmal_angle:
#     arr_x = np.column_stack((arr_x, dict_cut_no_mc_kmal_angle[key]))
    
# arr_x = arr_x[:,1:]
# x_mean = arr_x.mean(axis = 1)

# plt.figure()
# plt.plot(y_pred_mean, label = "pred")
# plt.plot(y_true_mean, label = "true")
# plt.plot(x_mean, label = "kma")
# plt.legend()

# plt.figure()
# plt.plot(y_pred_filtered_mean, label = "pred filtered")
# plt.plot(y_true_mean, label = "true")
# plt.plot(x_mean, label = "kma")
# plt.legend()
#%%

subBd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SB.pkl"
subB = op_pickle(subBd)

subCd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_thigh_SC.pkl"
subC = op_pickle(subCd)

tB = list(subB.items())
random.shuffle(tB)
testB = dict(tB)
pd_testB = pd.concat(testB, axis=0).reset_index(drop=True)
pd_testB = pd_testB.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])


tC = list(subC.items())
random.shuffle(tC)
testC = dict(tC)
pd_testC = pd.concat(testC, axis=0).reset_index(drop=True)
pd_testC = pd_testC.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank"])

Y_B = pd_testB.pop("no_mc_thigh_angle")
X_B = pd_testB

Y_C = pd_testC.pop("no_mc_thigh_angle")
X_C = pd_testC

predB = model.predict(X_B)
predC = model.predict(X_C)

scoreB = mean_squared_error(Y_B, predB)
scoreC = mean_squared_error(Y_C, predC)

plt.figure()
plt.title("sub B test")
plt.plot(predB, label = "prediction")
plt.plot(Y_B, label = "true")
plt.plot(X_B["no_mc_kmau_angle"])
plt.legend()

plt.figure()
plt.title("sub C test")
plt.plot(predC, label = "prediction")
plt.plot(Y_C, label = "true")
plt.plot(X_C["no_mc_kmau_angle"])
plt.legend()