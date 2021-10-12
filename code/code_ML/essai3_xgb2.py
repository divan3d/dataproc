# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:39:17 2021

@author: ksdiv
fait pseudo cross val parce que arrive pas a voir evl results avec gridsearchCV ds essai3
(je pense parce que a pas de eval set )
compliance (model 1)
pour choisir les features importants 
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
import statistics

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_knee_n_BC.pkl"
d = op_pickle(f)

#%%
number_of_folds = 5

#%% choose n estimators 

dict_of_res = {}
ds = list(d.items())

for x in range(number_of_folds):
    
    dict_of_res[x] = {}
    
    data_train, data_valid = train_test_split(ds, test_size=0.3)
    data_train = dict(data_train)
    data_valid = dict(data_valid)
    pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
    pd_valid = pd.concat(data_valid, axis = 0).reset_index(drop = True)
    
    # shank 
    # pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    # pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
    # Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
    # pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
    # X_train = pd_train
    
    # Y_valid = pd_valid["no_mc_shank_angle"] - pd_valid["no_mc_kmal_angle"]
    # pd_valid = pd_valid.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
    # X_valid = pd_valid
    
    #thigh 
    pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
    Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
    pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
    X_train = pd_train
    
    Y_valid = pd_valid["no_mc_shank_angle"] - pd_valid["no_mc_kmal_angle"]
    pd_valid = pd_valid.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
    X_valid = pd_valid

    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
    
    model = xgb.XGBRegressor()
    model.fit(X_train, Y_train, eval_set = eval_set, early_stopping_rounds = 10)
    
    Y_validtest = model.predict(X_valid)            
    validscore = mean_squared_error(Y_valid, Y_validtest)
  
    Y_traintest = model.predict(X_train)            
    trainscore = mean_squared_error(Y_train, Y_traintest)
    
    important_features = model.feature_importances_
    progress = model.evals_result()
    
    dict_of_res[x]["score validation"] = validscore
    dict_of_res[x]["score train"] = trainscore
    dict_of_res[x]["important features"] = important_features
    dict_of_res[x]["progress"] = progress
    dict_of_res[x]["best n iter"] = model.best_ntree_limit
    
    save_obj(dict_of_res, "shank_compliance_pscv_results_1.pkl")
    print ("finished one subject")
    
#%%

count = 0
tot_progress =[]
for sub in dict_of_res:
    tot_progress.append(dict_of_res[sub]["progress"])
    
    

plt.figure()
plt.grid()
for v in range(len(tot_progress)):
    plt.plot(tot_progress[v]['validation_0']["rmse"], label='train ')
    plt.plot(tot_progress[v]['validation_1']["rmse"], label='valid ')
plt.legend()
plt.title("shank compliance")
count = count + 1

#%% 
plt.figure()
plt.plot(Y_train)
plt.plot(Y_traintest)

plt.figure()
plt.plot(Y_valid)
plt.plot(Y_validtest)

#%% mean of important features to choose which ones to keep 

features_to_use = np.column_stack((dict_of_res[0]["important features"], dict_of_res[1]["important features"], 
                                   dict_of_res[2]["important features"],dict_of_res[3]["important features"],
                                   dict_of_res[4]["important features"]))

features_mean = np.mean(features_to_use, axis = 1)

list_keys = list(X_train)
b = np.column_stack((features_mean,list_keys))
c = b[b[:, 0].argsort()]
c = np.flip(c,0)
# order_feature = b[:,1]
# imp_feature = order_feature[-15:]
# not_imp_feature = order_feature[:-15]
# l_not_imp_feature = list(not_imp_feature)

# fait truc bizarre av echelle y !!
plt.figure()
plt.bar(range(len(c)), c[:,0])
plt.xticks(range(len(c)), c[:,1], rotation = 90)
plt.show()

c_val = c[:,0]
c_val = c_val.astype(float)
test = np.cumsum(c_val)
imp_feature = c[test[:]<= 0.95]
not_imp_feature = c[test[:]>0.95]

imp_feature90 = c[test[:]<= 0.9]
not_imp_feature90 = c[test[:]>0.9]

dict_thigh_features = {}
dict_thigh_features["keep 95%"] = imp_feature[:,1]
dict_thigh_features["remove 95%"] = not_imp_feature[:,1]
dict_thigh_features["keep 90%"] = imp_feature90[:,1]
dict_thigh_features["remove 90%"] = not_imp_feature90[:,1]

save_obj(dict_thigh_features, "features_for_shank.pkl")

#%%

plt.figure()
plt.plot(features_to_use[:,0], "o-", label = "1")
plt.plot(features_to_use[:,1], "o-", label = "2")
plt.plot(features_to_use[:,2], "o-", label = "3")
plt.plot(features_to_use[:,3], "o-", label = "4")
plt.plot(features_to_use[:,4], "o-", label = "5")
plt.legend()
plt.xticks(range(len(list_keys)), list_keys, rotation = 90)


#%%
dict_of_res = {}
ds = list(d.items())

for x in range(number_of_folds):
    
    dict_of_res[x] = {}
    
    data_train, data_valid = train_test_split(ds, test_size=0.3)
    data_train = dict(data_train)
    data_valid = dict(data_valid)
    pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
    pd_valid = pd.concat(data_valid, axis = 0).reset_index(drop = True)
    
    # shank 
    # pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    # pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
    # Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
    # pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
    # X_train = pd_train
    
    # Y_valid = pd_valid["no_mc_shank_angle"] - pd_valid["no_mc_kmal_angle"]
    # pd_valid = pd_valid.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
    # X_valid = pd_valid
    
    #thigh 
    pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
    Y_train = pd_train["no_mc_knee_angle"] - pd_train["no_mc_kma_rel_angle"]
    pd_train = pd_train.drop(columns = ["no_mc_knee_angle", "no_mc_kma_rel_angle"])
    X_train = pd_train
    
    Y_valid = pd_valid["no_mc_knee_angle"] - pd_valid["no_mc_kma_rel_angle"]
    pd_valid = pd_valid.drop(columns = ["no_mc_knee_angle", "no_mc_kma_rel_angle"])
    X_valid = pd_valid

    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
    
    model = xgb.XGBRegressor()
    model.fit(X_train, Y_train, eval_set = eval_set, early_stopping_rounds = 10)
    
    Y_validtest = model.predict(X_valid)            
    validscore = mean_squared_error(Y_valid, Y_validtest)
  
    Y_traintest = model.predict(X_train)            
    trainscore = mean_squared_error(Y_train, Y_traintest)
    
    important_features = model.feature_importances_
    progress = model.evals_result()
    
    dict_of_res[x]["score validation"] = validscore
    dict_of_res[x]["score train"] = trainscore
    dict_of_res[x]["important features"] = important_features
    dict_of_res[x]["progress"] = progress
    dict_of_res[x]["best n iter"] = model.best_ntree_limit
    
    save_obj(dict_of_res, "knee_compliance_pscv_results_1.pkl")
    print ("finished one subject")
    
#%%

count = 0
tot_progress =[]
for sub in dict_of_res:
    tot_progress.append(dict_of_res[sub]["progress"])
    
    

plt.figure()
plt.grid()
for v in range(len(tot_progress)):
    plt.plot(tot_progress[v]['validation_0']["rmse"], label='train ')
    plt.plot(tot_progress[v]['validation_1']["rmse"], label='valid ')
plt.legend()
plt.title("knee compliance")
count = count + 1

#%% 
plt.figure()
plt.plot(Y_train)
plt.plot(Y_traintest)

plt.figure()
plt.plot(Y_valid)
plt.plot(Y_validtest)

#%% mean of important features to choose which ones to keep 

features_to_use = np.column_stack((dict_of_res[0]["important features"], dict_of_res[1]["important features"], 
                                   dict_of_res[2]["important features"],dict_of_res[3]["important features"],
                                   dict_of_res[4]["important features"]))

features_mean = np.mean(features_to_use, axis = 1)

list_keys = list(X_train)
b = np.column_stack((features_mean,list_keys))
c = b[b[:, 0].argsort()]
c = np.flip(c,0)
# order_feature = b[:,1]
# imp_feature = order_feature[-15:]
# not_imp_feature = order_feature[:-15]
# l_not_imp_feature = list(not_imp_feature)

# fait truc bizarre av echelle y !!
plt.figure()
plt.bar(range(len(c)), c[:,0])
plt.xticks(range(len(c)), c[:,1], rotation = 90)
plt.show()

c_val = c[:,0]
c_val = c_val.astype(float)
test = np.cumsum(c_val)
imp_feature = c[test[:]<= 0.95]
not_imp_feature = c[test[:]>0.95]

imp_feature90 = c[test[:]<= 0.9]
not_imp_feature90 = c[test[:]>0.9]

dict_thigh_features = {}
dict_thigh_features["keep 95%"] = imp_feature[:,1]
dict_thigh_features["remove 95%"] = not_imp_feature[:,1]
dict_thigh_features["keep 90%"] = imp_feature90[:,1]
dict_thigh_features["remove 90%"] = not_imp_feature90[:,1]

save_obj(dict_thigh_features, "features_for_knee.pkl")

#%%

plt.figure()
plt.plot(features_to_use[:,0], "o-", label = "1")
plt.plot(features_to_use[:,1], "o-", label = "2")
plt.plot(features_to_use[:,2], "o-", label = "3")
plt.plot(features_to_use[:,3], "o-", label = "4")
plt.plot(features_to_use[:,4], "o-", label = "5")
plt.legend()
plt.xticks(range(len(list_keys)), list_keys, rotation = 90)
#%% choose max depth and min child weight 

# dict_of_res = {}

# max_depth_l = [4,5,6,7,9]
# min_child_weight_l = [1,3,5]

# ds = list(d.items())
# number_of_folds = 3

# for x in range(number_of_folds):
    
#     data_train, data_valid = train_test_split(ds, test_size=0.3)
#     data_train = dict(data_train)
#     data_valid = dict(data_valid)
#     pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
#     pd_valid = pd.concat(data_valid, axis = 0).reset_index(drop = True)
    
#     pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
#     pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
#     Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
#     pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
#     X_train = pd_train
    
#     Y_valid = pd_valid["no_mc_shank_angle"] - pd_valid["no_mc_kmal_angle"]
#     pd_valid = pd_valid.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
#     X_valid = pd_valid

#     eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
    
#     counter = 0
#     dict_of_res[x] = {}
    
#     for max_depth_val in max_depth_l:
#         for min_child_weight_val in min_child_weight_l:
    
#             model = xgb.XGBRegressor(n_estimators = 150, max_depth = max_depth_val, min_child_weight = min_child_weight_val)
#             model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)
            
#             Y_validtest = model.predict(X_valid)            
#             validscore = mean_squared_error(Y_valid, Y_validtest)
          
#             Y_traintest = model.predict(X_train)            
#             trainscore = mean_squared_error(Y_train, Y_traintest)
            
#             important_features = model.feature_importances_
#             progress = model.evals_result()
            
            
#             dict_of_res[x][counter] = {}
#             dict_of_res[x][counter]["score validation"] = validscore
#             dict_of_res[x][counter]["score train"] = trainscore
#             dict_of_res[x][counter]["important features"] = important_features
#             dict_of_res[x][counter]["progress"] = progress
#             dict_of_res[x][counter]["best n iter"] = model.best_ntree_limit
#             dict_of_res[x][counter]["max depth"] = max_depth_val
#             dict_of_res[x][counter]["min child weight"] = min_child_weight_val
            
#             save_obj(dict_of_res, "shank_compliance_pscv_results_2.pkl")
#             counter = counter + 1
            
#%% plot 

# veut faire la moyenne pour chaque cas 

# def get_mean_val(d_results):
#     val_score = [[] for i in range(len(d_results[0]))]
#     train_score = [[] for i in range(len(d_results[0]))]
#     depth_weight = []
    
#     for sub in d_results[0]:
#         depth_weight.append([d_results[0][sub]["max depth"],d_results[0][sub]["min child weight"]])
    
#     for sub in d_results:
#         for x in range( len(d_results[0])):
#             val_score[x].append(d_results[sub][x]["score validation"])
#             train_score[x].append(d_results[sub][x]["score train"])
            
            
#     val_mean = []
#     train_mean = []
#     for n in range(len(val_score)):
#         val_mean.append(statistics.mean(val_score[n]))
#         train_mean.append(statistics.mean(train_score[n]))
        
        
#     plt.figure()
#     plt.plot(val_mean, "o-")
#     plt.plot(train_mean, "x-")
#     plt.xticks(range(len(d_results[0])), depth_weight, rotation = 90)
#     return val_score, train_score


# v,t = get_mean_val(dict_of_res)
    



