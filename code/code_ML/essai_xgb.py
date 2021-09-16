# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 13:08:27 2021

@author: ksdiv
"""

import pseudo_train_test_split
import xgboost as xgb 
import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from sklearn.preprocessing import normalize, scale, MinMaxScaler

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
d_in = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/shank_gc.pkl"
dict_in = op_pickle(d_in)

# ça fait des bosses bizarres a cause de ou les gc sont coupées et misent ensemble 

XY_col_names = ["no_mc_shank_angle", "no_mc_kmal_angle", "Force"]
# atrainX, atrainY, atestX, atestY = pseudo_train_test_split.train_test_split_pseudo(dict_in, XY_col_names, 0.3, rdm_seed=45)
atrainX, atrainY, atestX, atestY, avalidX, avalidY = pseudo_train_test_split.train_test_valid_split_pseudo(dict_in, XY_col_names, 0.3, rdm_seed=45)


temp_train = np.column_stack((atrainX, atrainY))
temp_test = np.column_stack((atestX, atestY))
temp_valid = np.column_stack((avalidX, avalidY))

# train_norm = scale(temp_train)
# test_norm = scale(temp_test)

train_norm = temp_train
test_norm = temp_test
valid_norm = temp_valid

trainX_norm = train_norm[:,0:-1]
testX_norm = test_norm[:,0:-1]
validX_norm = valid_norm[:,0:-1]

trainY_norm = train_norm[:,-1]
testY_norm = test_norm[:,-1]
validY_norm = valid_norm[:,-1]


#%% xgb 

params = {}
params['objective'] = 'reg:squarederror'
params['eval_metric'] = 'rmse'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(trainX_norm, label = trainY_norm)
d_valid = xgb.DMatrix(validX_norm, label = validY_norm)
d_test = xgb.DMatrix(testX_norm, label = testY_norm)

evallist = [(d_valid, 'eval'), (d_train, 'train')]

#%%
num_round = 2000
tst = xgb.train(params, d_train, num_round, evallist, early_stopping_rounds=30)
#%%
# tst = xgb.train(params, d_train)

# !! enregistre PAS le meilleur modele, mais le dernier 
tst.save_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/first_try.model")

ypred = tst.predict(d_test)

#%%

plt.figure()
plt.plot(testY_norm, label = "real")
plt.plot(testX_norm[:,0], label = "x")
plt.plot(ypred, label = "pred")
plt.legend()