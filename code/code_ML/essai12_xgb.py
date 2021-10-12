# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:23:29 2021

@author: ksdiv
gamma =  min split loss 
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
import statistics

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#%%
# f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank.pkl"
# d = op_pickle(f)

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

l_drop_features_d = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_thigh.pkl"
l_drop_features = op_pickle(l_drop_features_d)

l_drop_features_k = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/features_for_knee.pkl"
l_drop_featuresk = op_pickle(l_drop_features_k)
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

#%%
# create train and validation set 
# subjects = ["SA", "SD", "SE","SF","SG","SH"]
subjects = {"SA" : 0, "SD" :1, "SE" :2,"SF" :3,"SG" :4,"SH" :5}

# choose one 
feat_drop = l_drop_features["remove 95%"]

number_of_folds = 3
# min_split_loss_l = [0, 0.1, 0.2, 0.3, 0.4]
# min_split_loss_l = [0,1,5,10]
min_split_loss_l = [0,10,100]
# reg_lambda_l = [1e-3, 0.1,1,100]

for key_sub in subjects :
    print("validation subject:")
    print(key_sub)
    sub_train = subjects.copy()
    del sub_train[key_sub]
    
    d_train = dict()
    for each_sub in sub_train:
        d_train.update(list_sub_dict[subjects[each_sub]])
        
    dict_val = dict(list_sub_dict[subjects[key_sub]])
     
    dico_results[key_sub] = {}

    for x in range(number_of_folds):   
        
        dico_results[key_sub][x] = {}

        ds = list(d_train.items())
        random.shuffle(ds)
        data_train, poubelle = train_test_split(ds, test_size=0.3)
        data_train = dict(data_train)
        pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
        
        d_val = list(dict_val.items())
        random.shuffle(d_val)
        data_val = dict(d_val)
        pd_valid = pd.concat(data_val, axis=0).reset_index(drop=True)
        
    
        pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_knee_angle", "no_mc_kma_rel_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
        Y_train = pd_train.pop("no_mc_shank_angle") 
        pd_train = pd_train.drop(columns = feat_drop)
        X_train = pd_train
        
        Y_valid = pd_valid.pop("no_mc_shank_angle") 
        pd_valid = pd_valid.drop(columns = feat_drop)
        X_valid = pd_valid
        
        eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
                  
        counter = 0
        for gamma_val in min_split_loss_l:
        # for lambda_val in reg_lambda_l:
            model = xgb.XGBRegressor(n_estimators = 100, max_depth = 3, min_child_weight = 1, min_split_loss = gamma_val)
            model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 10)
            
            print(model.best_ntree_limit)
            
            Y_validtest = model.predict(X_valid)            
            validscore = mean_squared_error(Y_valid, Y_validtest)
          
            Y_traintest = model.predict(X_train)            
            trainscore = mean_squared_error(Y_train, Y_traintest)
            
            important_features = model.feature_importances_
            progress = model.evals_result()
            
            dico_results[key_sub][x][counter] = {}
            dico_results[key_sub][x][counter]["score validation"] = validscore
            dico_results[key_sub][x][counter]["score train"] = trainscore
            dico_results[key_sub][x][counter]["important features"] = important_features
            dico_results[key_sub][x][counter]["gamma"] = gamma_val
            dico_results[key_sub][x][counter]["progress"] = progress
            dico_results[key_sub][x][counter]["best n iter"] = model.best_ntree_limit
            
            print("one iter finished")
            counter = counter + 1 
            save_obj(dico_results, "shank_error_gamma_3.pkl")


#%% knee
# create train and validation set 
# subjects = ["SA", "SD", "SE","SF","SG","SH"]
subjects = {"SA" : 0, "SD" :1, "SE" :2,"SF" :3,"SG" :4,"SH" :5}

# choose one 
feat_drop = l_drop_featuresk["remove 95%"]

number_of_folds = 3
# min_split_loss_l = [0, 0.1, 0.2, 0.3, 0.4]
# min_split_loss_l = [0,1,5,10]
min_split_loss_l = [0,10,100]
# reg_lambda_l = [1e-3, 0.1,1,100]

for key_sub in subjects :
    print("validation subject:")
    print(key_sub)
    sub_train = subjects.copy()
    del sub_train[key_sub]
    
    d_train = dict()
    for each_sub in sub_train:
        d_train.update(list_sub_dict[subjects[each_sub]])
        
    dict_val = dict(list_sub_dict[subjects[key_sub]])
     
    dico_results[key_sub] = {}

    for x in range(number_of_folds):   
        
        dico_results[key_sub][x] = {}

        ds = list(d_train.items())
        random.shuffle(ds)
        data_train, poubelle = train_test_split(ds, test_size=0.3)
        data_train = dict(data_train)
        pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
        
        d_val = list(dict_val.items())
        random.shuffle(d_val)
        data_val = dict(d_val)
        pd_valid = pd.concat(data_val, axis=0).reset_index(drop=True)
        
    
        pd_train = pd_train.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        pd_valid = pd_valid.drop(columns = ["t",  "no_mc_thigh_angle", "no_mc_kmau_angle","no_mc_shank_angle", "no_mc_kmal_angle", "vgrf", "vgrf1", "vgrf2", "GyroAThigh", "GyroAShank", "AlphaShank", "AlphaThigh"])
        
        Y_train = pd_train.pop("no_mc_knee_angle") 
        pd_train = pd_train.drop(columns = feat_drop)
        X_train = pd_train
        
        Y_valid = pd_valid.pop("no_mc_knee_angle") 
        pd_valid = pd_valid.drop(columns = feat_drop)
        X_valid = pd_valid
        
        eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
                  
        counter = 0
        for gamma_val in min_split_loss_l:
        # for lambda_val in reg_lambda_l:
            model = xgb.XGBRegressor(n_estimators = 100, max_depth = 3, min_child_weight = 1, min_split_loss = gamma_val)
            model.fit(X_train, Y_train, eval_set = eval_set, verbose = True, early_stopping_rounds = 10)
            
            print(model.best_ntree_limit)
            
            Y_validtest = model.predict(X_valid)            
            validscore = mean_squared_error(Y_valid, Y_validtest)
          
            Y_traintest = model.predict(X_train)            
            trainscore = mean_squared_error(Y_train, Y_traintest)
            
            important_features = model.feature_importances_
            progress = model.evals_result()
            
            dico_results[key_sub][x][counter] = {}
            dico_results[key_sub][x][counter]["score validation"] = validscore
            dico_results[key_sub][x][counter]["score train"] = trainscore
            dico_results[key_sub][x][counter]["important features"] = important_features
            dico_results[key_sub][x][counter]["gamma"] = gamma_val
            dico_results[key_sub][x][counter]["progress"] = progress
            dico_results[key_sub][x][counter]["best n iter"] = model.best_ntree_limit
            
            print("one iter finished")
            counter = counter + 1 
            save_obj(dico_results, "knee_error_gamma_3.pkl")


#%%
# dico_results = op_pickle(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/thigh_error_max_depth_min_child_weight_2.pkl")
colors = ["b", "g", "r", "c", "m", "k"]

#%% plot results 
def plot_val_and_train(d_results):
    count = 0
    plt.figure()
    
    sub_val_l = []
    sub_train_l = []
    
    tot_val_mean = np.empty(3)
    tot_train_mean = np.empty(3)
    
    for sub in d_results:
        sub_val = np.empty(3)
        sub_train = np.empty(3)
        score_val_t = []
        score_train_t  = []
        depth_weight = []
        
        for tt in d_results["SA"][0]:
            depth_weight.append(d_results["SA"][0][tt]["gamma"])
                
        for cv in d_results[sub]:
            for val in d_results[sub][cv]:
                score_val_t.append(d_results[sub][cv][val]["score validation"])
                score_train_t.append(d_results[sub][cv][val]["score train"])
            score_val = np.asarray(score_val_t)
            score_train = np.asarray(score_train_t)
            score_val_t = []
            score_train_t = []
            sub_val = np.column_stack((sub_val, score_val))
            sub_train = np.column_stack((sub_train, score_train))
                
        val_mean = np.mean(sub_val[:,1:], axis = 1)
        sub_val = np.column_stack((sub_val, val_mean))
        tot_val_mean = np.column_stack((tot_val_mean, val_mean))
        train_mean = np.mean(sub_train[:,1:], axis = 1)
        sub_train = np.column_stack((sub_train, train_mean))
        tot_train_mean = np.column_stack((tot_train_mean, train_mean))
        sub_val_l.append(sub_val[:,1:])
        sub_train_l.append(sub_train[:,1:])
                
                # depth_weight.append([d_results[sub][val]["max_depth"]])
            # depth_weight.append([d_results[sub][val]["col_sample"],d_results[sub][val]["subsample"]])
            
        
    
        plt.plot(val_mean, "o-", label = "score validation set, separate subject " + sub, color = colors[count])
        plt.plot(train_mean, "x-", label = "score on train set", color = colors[count])
        plt.legend()
        plt.title("effects of max depth and min child weight thigh error per subject")
        plt.xticks(range(len(depth_weight)), depth_weight, rotation = 90)
        count = count + 1
        
    # print(tot_val_mean)
    tot_val_mean = np.mean(tot_val_mean[:,1:], axis = 1)
    tot_train_mean = np.mean(tot_train_mean[:,1:], axis = 1)
    
    plt.figure()
    plt.plot(tot_train_mean, "x-", label = "train")
    plt.plot(tot_val_mean, "o-", label = "valid" )
    plt.title("effects of max depth and min child weight thigh error, mean")
    plt.xticks(range(len(depth_weight)), depth_weight, rotation = 90)
    plt.legend()
        
    return sub_val_l, sub_train_l


v,t = plot_val_and_train(dico_results)
#%%
u = op_pickle(r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/shank_error_gamma_3.pkl")
vu,tu = plot_val_and_train(u)
#%% plot results 
def plot_feature_imp(d_results):
    count = 0
    plt.figure()
        
    for sub in d_results:
        feature_imp = np.empty(39)
        feature_imp_t = []
                        
        for cv in d_results[sub]:
            for val in d_results[sub][cv]:
                plt.plot(d_results[sub][cv][val]["important features"], "o-", label =  sub, color = colors[count])
        #         feature_imp_t.append(d_results[sub][cv][val]["important features"])
        #         print(feature_imp_t)
        #     feature_val = np.asarray(feature_imp_t)
        #     feature_imp_t = []
        #     feature_imp = np.column_stack((feature_imp, feature_val))
                
        # feature_mean = np.mean(feature_imp[:,1:], axis = 1)
            
        # plt.plot(feature_mean, "o-", label =  sub, color = colors[count])
        plt.legend()
        plt.title("feature importance per sub")
        count = count + 1
        
        
    return 


# plot_feature_imp(dico_results)


#%%



def plot_progress(d_results):
    count = 0
    plt.figure()
        
    for sub in d_results:
                        
        for cv in d_results[sub]:
            for val in d_results[sub][cv]:
                plt.plot(d_results[sub][cv][val]["progress"]["validation_0"]["rmse"],  label =  "train", color = colors[count])
                plt.plot(d_results[sub][cv][val]["progress"]["validation_1"]["rmse"],  label =  "valid", color = colors[count])
        
        plt.legend()
        plt.title("progress")
        count = count + 1
    return

# plot_progress(dico_results)