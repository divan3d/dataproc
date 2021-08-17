# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:09:59 2021

@author: ksdiv

super slow

https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-3.html



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

XY_col_names = ["no_mc_thigh_angle", "no_mc_kmau_angle", "current_sent", "rolling_mean", "rolling_grad"]
trainX, trainY, testX, testY = pseudo_train_test_split.train_test_split_pseudo(thigh_data, XY_col_names, 0.3)

pipe_SVR = Pipeline([("scl", StandardScaler()), ("clf", SVR())])

# essayer en utilisant mean square error - plus " fort" pr les grosses erreurs 

param_C = [1, 10, 100, 500, 1000]
param_eps = [0.09, 0.1, 0.2]

grid_param = [{"clf__C": param_C, "clf__epsilon": param_eps}]

gs = GridSearchCV(estimator = pipe_SVR, param_grid = grid_param)

#%%

gs.fit(trainX, trainY)
print("best params: %s" %gs.best_params_)
print("best training accuracy: %3f" %gs.best_score_ )

# C = 1000, eps = 0.2 mais arrive pas les bosses 
#%%
y_pred = gs.predict(testX)

#%%
fig = plt.figure()
plt.plot(range(0,len(y_pred)), y_pred, color = "r")
plt.plot(range(0,len(y_pred)), testY, color = "g")
plt.plot(range(0,len(y_pred)), testX[:,0], color = "k")
