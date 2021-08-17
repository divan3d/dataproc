# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:29:20 2021

@author: ksdiv

preps thigh data for test svr 
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
import pseudo_train_test_split
from scipy import stats 
from random import shuffle, seed 
from sklearn.svm import SVR 
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize, scale, MinMaxScaler

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long_features.pkl"
thigh_data = op_pickle(thigh_data_file)


#%%
# faut mettre Y en tout premier pr col_names il me semble 
XY_col_names = ["no_mc_thigh_angle", "no_mc_kmau_angle", "current_sent", "rolling_mean", "rolling_grad", "current_change", "static_val"]
atrainX, atrainY, atestX, atestY = pseudo_train_test_split.train_test_split_pseudo(thigh_data, XY_col_names, 0.3, rdm_seed=45)

temp_train = np.column_stack((atrainX, atrainY))
temp_test = np.column_stack((atestX, atestY))

train_norm = scale(temp_train)
test_norm = scale(temp_test)

trainX_norm = train_norm[:,0:-1]
testX_norm = test_norm[:,0:-1]

trainY_norm = train_norm[:,-1]
testY_norm = test_norm[:,-1]

#%%

ng_XY_col_names = ["no_mc_thigh_angle", "no_mc_kmau_angle", "current_sent", "rolling_mean", "rolling_grad", "current_change"]
ngtrainX, ngtrainY, ngtestX, ngtestY = pseudo_train_test_split.train_test_split_pseudo(thigh_data, ng_XY_col_names, 0.3, rdm_seed=45)

ng_temp_train = np.column_stack((ngtrainX, ngtrainY))
ng_temp_test = np.column_stack((ngtestX, ngtestY))

ng_train_norm = scale(ng_temp_train)
ng_test_norm = scale(ng_temp_test)

ng_trainX_norm = ng_train_norm[:,0:-1]
ng_testX_norm = ng_test_norm[:,0:-1]

ng_trainY_norm = ng_train_norm[:,-1]
ng_testY_norm = ng_test_norm[:,-1]


#%% 

svr_rbf = SVR(kernel = "rbf")
reg_rbf = make_pipeline(svr_rbf)


#%% 

svr_rbf_norm = SVR(kernel = "rbf")
reg_rbf_norm = make_pipeline(svr_rbf_norm)
reg_rbf_norm.fit(trainX_norm,trainY_norm)
rbf_score_norm = reg_rbf_norm.score(testX_norm,testY_norm)

ynorm = reg_rbf_norm.predict(testX_norm)


#%%

svr_rbf_ng_norm = SVR(kernel = "rbf")
reg_rbf_ng_norm = make_pipeline(svr_rbf_ng_norm)
reg_rbf_ng_norm.fit(ng_trainX_norm,ng_trainY_norm)
rbf_score_ng_norm = reg_rbf_ng_norm.score(ng_testX_norm,ng_testY_norm)

y_ng_norm = reg_rbf_ng_norm.predict(ng_testX_norm)

#%%


fig = plt.figure()
plt.plot(range(0,len(ynorm)), ynorm, color = "r")
plt.plot(range(0,len(ynorm)), testY_norm, color = "g")
plt.plot(range(0,len(ynorm)), testX_norm[:,0], color = "k")
plt.title("with static info")


#%%
fig = plt.figure()
plt.plot(range(0,len(y_ng_norm)), y_ng_norm, color = "lightsteelblue")
plt.plot(range(0,len(y_ng_norm)), testY_norm, color = "hotpink")
plt.plot(range(0,len(y_ng_norm)), testX_norm[:,0], color = "blue")
plt.title("no static info ")

#%% 



# fait de la merde
# train_norm = normalize(temp_train)
# test_norm = normalize(temp_test)


# ca fait bien mais scale fait mieux 
# scaler = MinMaxScaler()
# scaler.fit(temp_train)
# train_norm = scaler.transform(temp_train)
# scaler.fit(temp_test)
# test_norm = scaler.transform(temp_test)