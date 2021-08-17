# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:56:11 2021

@author: ksdiv
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import statistics
import pseudo_train_test_split
from sklearn.svm import SVR 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import GridSearchCV

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long_features.pkl"
thigh_data = op_pickle(thigh_data_file)

thigh_data_s1 = thigh_data.pop("sub1")

XY_col_names = ["no_mc_thigh_angle", "no_mc_kmau_angle", "current_sent", "rolling_mean", "rolling_grad"]
trainX, trainY, testX, testY = pseudo_train_test_split.train_test_split_pseudo(thigh_data, XY_col_names, 0)

s1trainX, s1trainY, s1testX, s1testY = pseudo_train_test_split.train_test_split_pseudo(thigh_data_s1, XY_col_names, 0)

pipe_SVR = Pipeline([("scl", StandardScaler()), ("clf", SVR())])

param_C = [0.01, 0.1, 1, 10, 100]
param_eps = [0.09, 0.1, 0.2]

grid_param = [{"clf__C": param_C, "clf__epsilon": param_eps}]

gs = GridSearchCV(estimator = pipe_SVR, param_grid = grid_param)

#%%

gs.fit(trainX, trainY)
print("best params: %s" %gs.best_params_)
print("best training accuracy: %3f" %gs.best_score_ )

# permier essai : meilleur resultats: C = 100, epsilon = 0.1,  score 0.999771
# deuxieme essai : C = 1000, epsilon = 0.09, score 0.999784 
#%%
y_pred = gs.predict(s1trainX)

#%%
fig = plt.figure()
plt.plot(range(0,len(y_pred)), y_pred, color = "r")
plt.plot(range(0,len(y_pred)), s1trainY, color = "g")
plt.plot(range(0,len(y_pred)), s1trainX[:,0], color = "k")
plt.title("no grad ")