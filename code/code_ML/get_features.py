# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:48:31 2021

@author: ksdiv

get features from new experiment
for isometric : get dict of isometric features from get_features_isometric
adds these features to concentric

l'idee c'est que ça ouvre et envoie les trucs depuis ici, les fonctions seront stoquées
sur get_features_isometric par exemple
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
import os
import re
import glob

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def save_in_new_file(dirName, dict_data, name_file):  
    os.chdir(dir_name_sub)
            
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")    
        
    os.chdir(dirName)
    # print("Current working directory: {0}".format(os.getcwd()))

    save_obj(dict_data, name_file) 
        
subject = "SH"
dir_name_sub = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject
    
#%% cut and equalized gait cycles

#%% phase 

phase = np.ones(170)
phase[0:10] = 0
phase[20:30] = 2
phase[30:40] = 3
phase[40:50] = 4
phase[50:60] = 5
phase[60:70] = 6
phase[70:80] = 7
phase[80:90] = 8
phase[90:100] = 9
phase[100:110] = 10
phase[110:120] = 11
phase[120:130] = 12
phase[130:140] = 13
phase[140:150] = 14
phase[150:160] = 15
phase[160:170] = 16

#%%

def grad_d(d_in):
    """
    gets gradient 

    """ 
    grad = np.gradient(d_in)
    max_grad = max(grad)
    idx_max_grad = np.argmax(grad)
    min_grad = min(grad)
    idx_min_grad = np.argmin(grad)
    return grad, max_grad, idx_max_grad, min_grad, idx_min_grad


def get_encoder_dyn(d_in):
    enc = - d_in["HallSensor"]
    init = enc[0]
    max_val = max(enc)
    idx_min_grad = np.argmin(np.gradient(enc))
    d_in["dyn_eq_enc_init"] = init
    d_in["dyn_eq_enc_max"] = max_val
    d_in["dyn_eq_enc_idx_min_deriv"] = idx_min_grad
    return


def get_current_sent_dyn(d_in):
    min_val = min(d_in["current_read"][:110])
    offset = 0.2
    lim_val = min_val + offset
    nbr_pts = sum(d_in["current_read"] <= lim_val)
    d_in["dyn_eq_current_read_min"] = min_val
    d_in["dyn_eq_current_read_width_neg"] = nbr_pts
    # not current mais bon
    d_in["dyn_eq_kmal_max"] = max(d_in["no_mc_kmal_angle"])
    d_in["dyn_eq_kmal_idx_max"] = np.argmax(d_in["no_mc_kmal_angle"])
    d_in["dyn_eq_kmal_min"] = min(d_in["no_mc_kmal_angle"])
    d_in["dyn_eq_kmal_idx_min"] = np.argmin(d_in["no_mc_kmal_angle"])
    d_in["dyn_eq_gyro_s_deriv"] = np.gradient(d_in["GyroCShank"])
    d_in["dyn_eq_gyro_t_deriv"] = np.gradient(d_in["GyroCThigh"])
    d_in["dyn_eq_phase"] = phase
    d_in["dyn_eq_force_on_off"] = d_in["Force"] > 2
    d_in["dyn_eq_force_on_off"] = d_in["dyn_eq_force_on_off"].astype(int)
    return

#%%

# for shank_f in shank_file:
#     eq_shank = op_pickle(dir_eq_shank + "//" + shank_f)
#     for key in eq_shank:


#%% features

# sub info 
dir_sub_info = dir_name_sub + "//" + subject + "_features_sub_info/" + subject + "_features_subject_info.pkl"
features_sub_info = op_pickle(dir_sub_info)

# uncut dynamic 
# dir_features_uncut_dyn = dir_name_sub + "//" + subject + "_features_dyn_uncut"

# isometric
dir_features_iso = dir_name_sub + "//" + subject + "_features_ISO"

#%% concentric


#%% 

def add_iso_features_to_concentric(features, concentric_df):
    for key in features: 
        concentric_df[key] = features[key]
    return

def add_to_dict_con(concentric_dict):
    for key in concentric_dict:
        #sub info
        add_iso_features_to_concentric(features_sub_info, concentric_dict[key])
        # get rid of cadence since added "InitialLength"
        # add_iso_features_to_concentric(uncut_dyn_features, concentric_dict[key])
        add_iso_features_to_concentric(iso_features, concentric_dict[key])
        get_encoder_dyn(concentric_dict[key])
        get_current_sent_dyn(concentric_dict[key])
    return

# add_iso_features_to_concentric(iso1_features, con1)
# add_iso_features_to_concentric(iso2_features, con2)

# add_to_dict_con(iso1_features, c2)

#%%

# test_dir = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_EQ_GC_shank/SA_FL3_cut_shank_2.pkl"
# test = op_pickle(test_dir)

# add_to_dict_con(test, test)

#%% iso et dyn uncut faut aller chercher le bon fichier a chaque truc 

list_fl_files = glob.glob(dir_name_sub + "\\" + subject + "_FL*")

for fl_by_subject in list_fl_files:
    fl = fl_by_subject[48:]
    #get iso features
    features_iso_path = glob.glob(dir_features_iso + "/features_" + subject + "_FL" + fl + "_cut_Isometric*")
    for features_iso in features_iso_path:
        exp_nbr = features_iso[-5]
        iso_features = op_pickle(features_iso)        
        # get concentric equal dynamic data
        dir_eq_shank = dir_name_sub + "//" + subject + "_EQ_GC_shank/" + subject + "_FL" + fl + "_cut_shank_" + exp_nbr + ".pkl"
        dir_eq_thigh = dir_name_sub + "//" + subject + "_EQ_GC_thigh/" + subject + "_FL" + fl + "_cut_thigh_" + exp_nbr + ".pkl"
        conc_shank = op_pickle(dir_eq_shank)
        conc_thigh = op_pickle(dir_eq_thigh)
        # get uncut dyn features 
        # dir_uncut_features = dir_features_uncut_dyn + "/features_" + subject + "_FL" + fl + "_Concentric" + exp_nbr + ".pkl"
        # uncut_dyn_features = op_pickle(dir_uncut_features)
        #add features 
        add_to_dict_con(conc_shank)
        add_to_dict_con(conc_thigh)
        save_in_new_file(subject + "_features", conc_shank, "features_fin_shank_" + subject + "_FL" + fl + "_exp" + exp_nbr + ".pkl")
        save_in_new_file(subject + "_features", conc_thigh, "features_fin_thigh_" + subject + "_FL" + fl + "_exp" + exp_nbr + ".pkl")


        
    




