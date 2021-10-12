# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:07:57 2021

@author: ksdiv
change learning rate
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

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

subAd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SA.pkl"
subA = op_pickle(subAd)

subDd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SD.pkl"
subD = op_pickle(subDd)

subEd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SE.pkl"
subE = op_pickle(subEd)

subFd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SF.pkl"
subF = op_pickle(subFd)

subGd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SG.pkl"
subG = op_pickle(subGd)

subHd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SH.pkl"
subH = op_pickle(subHd)

subBd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SB.pkl"
subB = op_pickle(subBd)

subCd = r"E:/ETHZ/mast_sem_IV/pdm/code/dict_gait_cycle_knee_SC.pkl"
subC = op_pickle(subCd)

l_drop_features_d = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_shank.pkl"
l_drop_features = op_pickle(l_drop_features_d)

l_drop_features_k = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_knee.pkl"
l_drop_featuresk = op_pickle(l_drop_features_k)

l_drop_features_t = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_thigh.pkl"
l_drop_featurest = op_pickle(l_drop_features_t)
#%%
dico_results = {}

#%%

list_sub_dict = []
list_sub_dict.append(subA)
list_sub_dict.append(subD)
list_sub_dict.append(subE)
list_sub_dict.append(subF)
list_sub_dict.append(subG)
list_sub_dict.append(subH)
subjects = {"SA" : 0, "SD" :1, "SE" :2,"SF" :3,"SG" :4,"SH" :5}

#%%
# create train and validation set 
# subjects = ["SA", "SD", "SE","SF","SG","SH"]


# subjects = {"SA" : 0, "SD" :1, "SE" :2,"SF" :3,"SG" :4,"SH" :5}
# feat_drop = l_drop_features["remove 95%"]


# dico_results = {}


# learning_rate_l = [0.3, 0.1, 0.05, 0.01]

# for key_sub in subjects :
#     print("validation subject:")
#     print(key_sub)
#     sub_train = subjects.copy()
#     del sub_train[key_sub]
    
#     d_train = dict()
#     for each_sub in sub_train:
#         d_train.update(list_sub_dict[subjects[each_sub]])
        
#     dict_val = dict(list_sub_dict[subjects[key_sub]])

#     ds = list(d_train.items())
#     random.shuffle(ds)
#     data_train = dict(ds)
#     pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
    
#     d_val = list(dict_val.items())
#     random.shuffle(d_val)
#     data_val = dict(d_val)
#     pd_valid = pd.concat(data_val, axis=0).reset_index(drop=True)

#     pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
#     pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    
#     Y_train = pd_train.pop("no_mc_shank_angle") 
#     # Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
#     # pd_train = pd_train.drop(columns = ["no_mc_knee_angle", "no_mc_kma_rel_angle"])
#     # pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
#     pd_train = pd_train.drop(columns = feat_drop)
#     X_train = pd_train
    
#     Y_valid = pd_valid.pop("no_mc_shank_angle") 
#     pd_valid = pd_valid.drop(columns = feat_drop)
#     X_valid = pd_valid
    
#     # plt.figure()
#     # plt.plot(Y_train)
    
#     eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

#     dico_results[key_sub] = {}
#     counter = 0
#     for learning_rate_val in learning_rate_l:
#         if learning_rate_val == 0.05 or learning_rate_val == 0.01:
#             model = xgb.XGBRegressor(learning_rate = learning_rate_val, n_estimators = 1000, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_alpha = 1, subsample = 0.9, colsample_bytree = 0.9, reg_lambda = 0.1)
#             model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 50)
#         if learning_rate_val == 0.1 or learning_rate_val == 0.3:
#             model = xgb.XGBRegressor(learning_rate = learning_rate_val, n_estimators = 100, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_alpha = 1, subsample = 0.9, colsample_bytree = 0.9, reg_lambda = 0.1)
#             model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 10)
        
#         Y_validtest = model.predict(X_valid)            
#         validscore = mean_squared_error(Y_valid, Y_validtest)
      
#         Y_traintest = model.predict(X_train)            
#         trainscore = mean_squared_error(Y_train, Y_traintest)
        
#         important_features = model.feature_importances_
#         progress = model.evals_result()
        
#         dico_results[key_sub][counter] = {}
#         dico_results[key_sub][counter]["learning rate"] = learning_rate_val
#         dico_results[key_sub][counter]["score validation"] = validscore
#         dico_results[key_sub][counter]["score train"] = trainscore
#         dico_results[key_sub][counter]["important features"] = important_features
#         dico_results[key_sub][counter]["progress"] = progress
#         dico_results[key_sub][counter]["best n iter"] = model.best_ntree_limit
#         counter = counter + 1
#         save_obj(dico_results, "shank_error_learning_rate_1.pkl")
        
        #%%

# feat_drop = l_drop_featuresk["remove 95%"]


# dico_results = {}


# learning_rate_l = [0.3, 0.1, 0.05, 0.01]

# for key_sub in subjects :
#     print("validation subject:")
#     print(key_sub)
#     sub_train = subjects.copy()
#     del sub_train[key_sub]
    
#     d_train = dict()
#     for each_sub in sub_train:
#         d_train.update(list_sub_dict[subjects[each_sub]])
        
#     dict_val = dict(list_sub_dict[subjects[key_sub]])

#     ds = list(d_train.items())
#     random.shuffle(ds)
#     data_train = dict(ds)
#     pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
    
#     d_val = list(dict_val.items())
#     random.shuffle(d_val)
#     data_val = dict(d_val)
#     pd_valid = pd.concat(data_val, axis=0).reset_index(drop=True)

#     pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
#     pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
    
#     Y_train = pd_train.pop("no_mc_knee_angle") 
#     # Y_train = pd_train["no_mc_shank_angle"] - pd_train["no_mc_kmal_angle"]
#     # pd_train = pd_train.drop(columns = ["no_mc_knee_angle", "no_mc_kma_rel_angle"])
#     # pd_train = pd_train.drop(columns = ["no_mc_shank_angle", "no_mc_kmal_angle"])
#     pd_train = pd_train.drop(columns = feat_drop)
#     X_train = pd_train
    
#     Y_valid = pd_valid.pop("no_mc_knee_angle") 
#     pd_valid = pd_valid.drop(columns = feat_drop)
#     X_valid = pd_valid
    
#     # plt.figure()
#     # plt.plot(Y_train)
    
#     eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

#     dico_results[key_sub] = {}
#     counter = 0
#     for learning_rate_val in learning_rate_l:
#         if learning_rate_val == 0.05 or learning_rate_val == 0.01:
#             model = xgb.XGBRegressor(learning_rate = learning_rate_val, n_estimators = 1000, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_lambda = 100, subsample = 0.9, colsample_bytree = 0.7)
#             model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 50)
#         if learning_rate_val == 0.1 or learning_rate_val == 0.3:
#             model = xgb.XGBRegressor(learning_rate = learning_rate_val, n_estimators = 100, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_lambda = 100, subsample = 0.9, colsample_bytree = 0.7)
#             model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 10)
        
#         Y_validtest = model.predict(X_valid)            
#         validscore = mean_squared_error(Y_valid, Y_validtest)
      
#         Y_traintest = model.predict(X_train)            
#         trainscore = mean_squared_error(Y_train, Y_traintest)
        
#         important_features = model.feature_importances_
#         progress = model.evals_result()
        
#         dico_results[key_sub][counter] = {}
#         dico_results[key_sub][counter]["learning rate"] = learning_rate_val
#         dico_results[key_sub][counter]["score validation"] = validscore
#         dico_results[key_sub][counter]["score train"] = trainscore
#         dico_results[key_sub][counter]["important features"] = important_features
#         dico_results[key_sub][counter]["progress"] = progress
#         dico_results[key_sub][counter]["best n iter"] = model.best_ntree_limit
#         counter = counter + 1
#         save_obj(dico_results, "knee_error_learning_rate_1.pkl")

#%%
# Y_pred = model.predict(X_test)

# score = mean_squared_error(Y_test, Y_pred)

# list_keys = list(X_valid)

# plt.figure()
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plt.xticks(range(len(model.feature_importances_)), list_keys, rotation = 90)
# plt.show()

# #%%
# a = model.feature_importances_
# a = np.column_stack((a,list_keys))
# a[a[:, 0].argsort()]

# order_feature = a[:,1]
# imp_feature = order_feature[-15:]
# not_imp_feature = order_feature[:-15]
# l_not_imp_feature = list(not_imp_feature)

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

#%% train model in full and test on sub C and D 
feat_dropt = l_drop_featurest["remove 95%"]
# thigh

        
d_f_train = dict()
for each_sub in subjects:
    d_f_train.update(list_sub_dict[subjects[each_sub]])
d_fs = list(d_f_train.items())
random.shuffle(d_fs)
data_f_train = dict(d_fs)
pd_f_train = pd.concat(data_f_train, axis=0).reset_index(drop=True)

pd_f_train = pd_f_train.drop(columns = ["t",  "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_f_train = pd_f_train.pop("no_mc_thigh_angle") 
pd_f_train = pd_f_train.drop(columns = feat_dropt)
X_f_train = pd_f_train

fullmodel = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 100, max_depth = 8, reg_alpha = 0.1, subsample = 0.8, colsample_bytree = 0.9, min_child_weight = 1, min_split_loss = 0.1 )
fullmodel.fit(X_f_train, Y_f_train, verbose = True)
fullmodel.save_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/thigh_1.model")

d_testC = list(subC.items())
random.shuffle(d_testC)
test_C = dict(d_testC)
pd_testC = pd.concat(test_C, axis = 0).reset_index(drop= True)
pd_testC = pd_testC.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_testC = pd_testC.pop("no_mc_thigh_angle") 
pd_testC = pd_testC.drop(columns = feat_dropt)

d_testB = list(subB.items())
random.shuffle(d_testB)
test_B = dict(d_testB)
pd_testB = pd.concat(test_B, axis = 0).reset_index(drop= True)
pd_testB = pd_testB.drop(columns = ["t", "no_mc_shank_angle", "no_mc_kmal_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_testB = pd_testB.pop("no_mc_thigh_angle") 
pd_testB = pd_testB.drop(columns = feat_dropt)

predC = fullmodel.predict(pd_testC)            
Cscore_thigh = mean_squared_error(predC, Y_testC)

predB = fullmodel.predict(pd_testB)            
Bscore_thigh = mean_squared_error(predB, Y_testB)

plt.figure()
plt.plot(Y_testC, label = "Y true")
plt.plot(pd_testC["no_mc_kmau_angle"], label = "x input")
plt.plot(predC, label = "output")
plt.legend()
plt.title("thigh sub C")

plt.figure()
plt.plot(Y_testB, label = "Y true")
plt.plot(pd_testB["no_mc_kmau_angle"], label = "x input")
plt.plot(predB, label = "output")
plt.legend()
plt.title("thigh sub B")

#%%
# shank
feat_drops = l_drop_features["remove 95%"]
        
d_f_train = dict()
for each_sub in subjects:
    d_f_train.update(list_sub_dict[subjects[each_sub]])
d_fs = list(d_f_train.items())
random.shuffle(d_fs)
data_s_train = dict(d_fs)
pd_s_train = pd.concat(data_s_train, axis=0).reset_index(drop=True)

pd_s_train = pd_s_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])

Y_s_train = pd_s_train.pop("no_mc_shank_angle") 
pd_s_train = pd_s_train.drop(columns = feat_drops)
X_s_train = pd_s_train


model_shank = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 150, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_alpha = 1, subsample = 0.9, colsample_bytree = 0.9, reg_lambda = 0.1)
model_shank.fit(X_s_train, Y_s_train, verbose = True)
model_shank.save_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/shank_1.model")

d_testC = list(subC.items())
random.shuffle(d_testC)
test_C = dict(d_testC)
pd_testC = pd.concat(test_C, axis = 0).reset_index(drop= True)
pd_testC = pd_testC.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_testC = pd_testC.pop("no_mc_shank_angle") 
pd_testC = pd_testC.drop(columns = feat_drops)

d_testB = list(subB.items())
random.shuffle(d_testB)
test_B = dict(d_testB)
pd_testB = pd.concat(test_B, axis = 0).reset_index(drop= True)
pd_testB = pd_testB.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_testB = pd_testB.pop("no_mc_shank_angle") 
pd_testB = pd_testB.drop(columns = feat_drops)

predC = model_shank.predict(pd_testC)            
Cscore_shank = mean_squared_error(predC, Y_testC)

predB = model_shank.predict(pd_testB)            
Bscore_shank = mean_squared_error(predB, Y_testB)

plt.figure()
plt.plot(Y_testC, label = "Y true")
plt.plot(pd_testC["no_mc_kmal_angle"], label = "x input")
plt.plot(predC, label = "output")
plt.legend()
plt.title("shank sub C")

plt.figure()
plt.plot(Y_testB, label = "Y true")
plt.plot(pd_testB["no_mc_kmal_angle"], label = "x input")
plt.plot(predB, label = "output")
plt.legend()
plt.title("shank sub B")

#%%
# knee
feat_dropk = l_drop_featuresk["remove 95%"]
        
d_f_train = dict()
for each_sub in subjects:
    d_f_train.update(list_sub_dict[subjects[each_sub]])
d_fs = list(d_f_train.items())
random.shuffle(d_fs)
data_k_train = dict(d_fs)
pd_k_train = pd.concat(data_k_train, axis=0).reset_index(drop=True)

pd_k_train = pd_k_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])

Y_k_train = pd_k_train.pop("no_mc_knee_angle") 
pd_k_train = pd_k_train.drop(columns = feat_dropk)
X_k_train = pd_k_train

model_knee = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 1000, max_depth = 3, min_child_weight = 1, min_split_loss = 0.1, reg_lambda = 100, subsample = 0.9, colsample_bytree = 0.7)
model_knee.fit(X_k_train, Y_k_train,  verbose = True)
model_knee.save_model(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/knee_1.model")

d_testC = list(subC.items())
random.shuffle(d_testC)
test_C = dict(d_testC)
pd_testC = pd.concat(test_C, axis = 0).reset_index(drop= True)
pd_testC = pd_testC.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_testC = pd_testC.pop("no_mc_knee_angle") 
pd_testC = pd_testC.drop(columns = feat_dropk)

d_testB = list(subB.items())
random.shuffle(d_testB)
test_B = dict(d_testB)
pd_testB = pd.concat(test_B, axis = 0).reset_index(drop= True)
pd_testB = pd_testB.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
Y_testB = pd_testB.pop("no_mc_knee_angle") 
pd_testB = pd_testB.drop(columns = feat_dropk)

predC = model_knee.predict(pd_testC)            
Cscore_knee = mean_squared_error(predC, Y_testC)

predB = model_knee.predict(pd_testB)            
Bscore_knee = mean_squared_error(predB, Y_testB)

plt.figure()
plt.plot(Y_testC, label = "Y true")
plt.plot(pd_testC["no_mc_kma_rel_angle"], label = "x input")
plt.plot(predC, label = "output")
plt.legend()
plt.title("knee sub C")

plt.figure()
plt.plot(Y_testB, label = "Y true")
plt.plot(pd_testB["no_mc_kma_rel_angle"], label = "x input")
plt.plot(predB, label = "output")
plt.legend()
plt.title("knee sub B")

#%% feature importance 

shank_features = model_shank.feature_importances_
thigh_features = fullmodel.feature_importances_
knee_features = model_knee.feature_importances_


def plot_feat(labels, features,name):
    plt.figure()
    plt.bar(range(len(features)), features)
    plt.xticks(range(len(features)), labels, rotation = 90)
    plt.title(name)
    plt.show()  
    return

label_knee = list(pd_k_train)
label_shank = list(pd_s_train)
label_thigh = list(pd_f_train)

plot_feat(label_knee, knee_features, "knee features")
plot_feat(label_shank, shank_features, "shank features")
plot_feat(label_thigh, thigh_features, "thigh features")

#%%
# b_shank = np.column_stack((shank_features,label_shank))
# c_shank = b_shank[b_shank[:, 0].argsort()]
# c_shank = np.flip(c_shank,0)

# #%%
# b_thigh = np.column_stack((thigh_features,label_thigh))
# c_thigh = b_thigh[b_thigh[:, 0].argsort()]
# c_thigh = np.flip(c_thigh,0)

# #%%
# b_knee = np.column_stack((knee_features,label_knee))
# c_knee = b_knee[b_knee[:, 0].argsort()]
# c_knee = np.flip(c_knee,0)