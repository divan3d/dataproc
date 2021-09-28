# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:00:59 2021

@author: ksdiv
trying out which subjects left out get best results
"""

import pickle
import glob
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb 
from numpy import random
import pandas as pd

#%%
def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#%%

def create_dict_test_train(subjects_interest):
    dict_shank = {}
    counter = 0
    for sub in subjects_interest:
        # temp_dir = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + sub + "//" + sub + "_EQ_GC_shank"
        temp_dir = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + sub + "//" + sub + "_features"
        # temp_list = os.listdir(temp_dir)
        temp_list = glob.glob(temp_dir + "/features_fin_shank*")
        for file in temp_list :
            # whole_dir = temp_dir + "//" + file
            # dict_gc = op_pickle(whole_dir)
            dict_gc = op_pickle(file)
            for key in dict_gc:
                dict_shank[str(counter)] = dict_gc[key]
                counter = counter + 1
    return dict_shank

#%% 

results = {}

subjects = ["SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH"]
comb = list(combinations(subjects, 6))


for comb_nbr in range(len(comb)) :
    sub_train = list(comb[comb_nbr])
    sub_test = subjects.copy()
    for i in range(len(sub_train)):
        sub_test.remove(sub_train[i])
        
    results[comb_nbr] = {}
    results[comb_nbr]["trained"] = sub_train
    results[comb_nbr]["tested"] = sub_test    
    
    dict_train = create_dict_test_train(sub_train)
    dict_test = create_dict_test_train(sub_test)
        

    ds = list(dict_train.items())
    random.shuffle(ds)
    data_train, data_valid = train_test_split(ds, test_size=0.3, random_state=42)
    data_train = dict(data_train)
    data_valid = dict(data_valid)
    pd_train = pd.concat(data_train, axis=0).reset_index(drop=True)
    pd_valid = pd.concat(data_valid, axis = 0).reset_index(drop = True)
    
    tests = list(dict_test.items())
    random.shuffle(tests)
    test_val = dict(tests)
    pd_test = pd.concat(test_val, axis = 0).reset_index(drop= True)


    pd_train = pd_train.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
    pd_valid = pd_valid.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
    pd_test = pd_test.drop(columns = ["t", "no_mc_thigh_angle", "no_mc_kmau_angle", "vgrf", "vgrf1", "vgrf2"])
    
    Y_train = pd_train.pop("no_mc_shank_angle")
    X_train = pd_train
    
    Y_valid = pd_valid.pop("no_mc_shank_angle")
    X_valid = pd_valid
    
    Y_test = pd_test.pop("no_mc_shank_angle")
    X_test = pd_test
    
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
    
    model = xgb.XGBRegressor()
    model.fit(X_train, Y_train, eval_set = eval_set, verbose = True)
    
    
    Y_pred = model.predict(X_test)
    
    score = mean_squared_error(Y_test, Y_pred)
    results[comb_nbr]["MSE"] = score
    print("finished one iteration, score:")
    print(score)
    print("with test subjects") 
    print(sub_test)


    save_obj(results, "results_combination.pkl")

#%%

import matplotlib.pyplot as plt 

test_subjects = []
results_test = []
for x in results :
    test_subjects.append(results[x]["tested"])
    results_test.append(results[x]["MSE"])
    
#%%

plt.figure()
plt.bar(range(28), results_test)
plt.xticks(range(28), test_subjects, rotation = 90)
plt.show()


