# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:41:42 2021

@author: ksdiv
"""

# pr suivre la logique, faudrait créer un fichier extracted_data/SUB/SUB_features_sub_info

import pickle
import os

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def save_in_new_file(dir_name_sub, dirName, dict_data, name_file):  
    os.chdir(dir_name_sub)
            
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")    
        
    os.chdir(dirName)
    # print("Current working directory: {0}".format(os.getcwd()))

    save_obj(dict_data, name_file) 
    
    
#%%

SA_info = {}
SA_info["height"] = 1.63
SA_info["weight"] = 55.9
SA_info["BMI"] = 55.9/(1.63 * 1.63)

SB_info = {}
SB_info["height"] = 1.78
SB_info["weight"] = 76.3
SB_info["BMI"] = 76.3/(1.78*1.78)

SC_info = {}
SC_info["height"] = 1.63
SC_info["weight"] = 52
SC_info["BMI"] = 52/(1.63*1.63)

SD_info = {}
SD_info["height"] = 1.7
SD_info["weight"] = 60
SD_info["BMI"] = 60/(1.7*1.7)

SE_info = {}
SE_info["height"] = 1.62
SE_info["weight"] = 53
SE_info["BMI"] = 53/(1.62*1.62)

SF_info= {}
SF_info["height"] = 1.68
SF_info["weight"] = 51
SF_info["BMI"] = 51/(1.68*1.68)

SG_info = {}
SG_info["height"] = 1.95
SG_info["weight"] = 85
SG_info["BMI"] = 85/(1.95*1.95)

SH_info = {}
SH_info["height"] = 1.77
SH_info["weight"] = 77
SH_info["BMI"] = 77/(1.77*1.77)

big_list = [SA_info, SB_info, SC_info, SD_info, SE_info, SF_info, SG_info, SH_info]


#%%

subjects = ["SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH"]
counter = 0
for subject in subjects:
    dir_name_subb = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject
    name = subject + "_features_subject_info.pkl"
    name_dict = subject + "_features_sub_info"
    save_in_new_file(dir_name_subb, name_dict, big_list[counter], name)
    counter = counter + 1
    
#%%

# t = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_features_sub_info/SA_features_subject_info"
# tt = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_features_ISO/features_SA_FL1_cut_Isometric2.pkl"

# ttt = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SF/SF_features_sub_info/SF_features_subject_info"

# def op_pickle(file):
#     with open(file,'rb') as fileopen:
#         data=pickle.load(fileopen)
#         return data
    
# d = op_pickle(t)
# dd = op_pickle(tt)

# ddd = op_pickle(ttt)