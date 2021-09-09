# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:10:32 2021

@author: ksdiv
get MSE from all the data we have
want a dict with name of folder, MSE for that folder 
"""

import pickle
import os
import statistics
from sklearn.metrics import mean_squared_error

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
subjects = ["SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH"]

whole_dict_shank = {}
for sub in subjects: 

    file_name = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + sub + "/" + sub + "_CUT_EGC/shank" 
    
    list_files = os.listdir(file_name)
    
    whole_dict_shank[sub + "_dict"] = {}
    for f_name in list_files:
        file_now = op_pickle(file_name + "/" + f_name)
        temp_list_shank = []
        for key in file_now:
            temp_list_shank.append(mean_squared_error(file_now[key]["no_mc_shank_angle"], file_now[key]["no_mc_kmal_angle"]))
        whole_dict_shank[sub + "_dict"][f_name + "shank_err"] = statistics.mean(temp_list_shank)
    
    
whole_dict_thigh = {}
for sub in subjects: 

    file_name = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + subjects[0] + "/" + subjects[0] + "_CUT_EGC/thigh" 
    
    list_files = os.listdir(file_name)
    
    whole_dict_thigh[sub + "_dict"] = {}
    for f_name in list_files:
        file_now = op_pickle(file_name + "/" + f_name)
        temp_list_thigh = []
        for key in file_now:
            temp_list_thigh.append(mean_squared_error(file_now[key]["no_mc_thigh_angle"], file_now[key]["no_mc_kmau_angle"]))
        whole_dict_thigh[sub + "_dict"][f_name + "thigh_err"] = statistics.mean(temp_list_thigh)
    
#%%

list_shank_mean = []
for key in whole_dict_shank:
    for s_key in whole_dict_shank[key]:
        list_shank_mean.append(whole_dict_shank[key][s_key])
mean_shank = statistics.mean(list_shank_mean)

#mean shank = 19.1948 (au carré 4.poussieres)
    
#%%
list_thigh_mean = []
for key in whole_dict_thigh:
    for s_key in whole_dict_thigh[key]:
        list_thigh_mean.append(whole_dict_thigh[key][s_key])
mean_thigh = statistics.mean(list_thigh_mean)

# mean_thigh = 17.367 (a mettre au carré)