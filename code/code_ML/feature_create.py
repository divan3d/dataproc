# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:31:43 2021

@author: ksdiv

feature engineering 
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long.pkl"
thigh_data = op_pickle(thigh_data_file)

#%% 
# when scale data, want everything to be above 0 - at the end we only care 
# about the difference btwn the values so just add csts 

# ça n'a pas besoin d'etre fait ici.. peut etre dans data processing functions
# peut l'ecrire ici et le déplacer ds DataProcessingFunction plus tard

# pas necessaire ??!!?? # 

val_current = 15
val_angles = 40 


def add_val_current(val_current, dict_in):
    for key in dict_in:
        dict_in[key]["current_sent"] = dict_in[key]["current_sent"].add(val_current)
    return

def add_val_thigh(val_angles, dict_in):
    for key in dict_in:
        dict_in[key]["no_mc_thigh_angle"] = dict_in[key]["no_mc_thigh_angle"].add(val_angles)
        dict_in[key]["no_mc_kmau_angle"] = dict_in[key]["no_mc_kmau_angle"].add(val_angles)
    return

# for key in thigh_data:
#     add_val_current(thigh_data[key], val_current)
#     add_val_thigh(thigh_data[key], val_angles)

#%% 
def add_features_rolling_mean_wdw(dict_in, wdw_size):
    for key in dict_in:
        # mean 
        dict_in[key]["rolling_mean"] = dict_in[key]["no_mc_kmau_angle"].rolling(wdw_size).mean()
        og = dict_in[key]["no_mc_kmau_angle"][0:wdw_size].copy()
        # lance un warning 
        dict_in[key]["rolling_mean"][0:wdw_size] = og 
        
    return 


def add_features_rolling_grad_wdw(dict_in, wdw_size):
    for key in dict_in:           
        # grad 
        # impression que faut rendre la fenetre plus petite, sinon ça verra pas quand ça remonte 
        dict_in[key]["rolling_grad"] = np.gradient(dict_in[key]["no_mc_kmau_angle"].rolling(wdw_size).mean())
        # fonctionne pas forcément bien - surtout que début correspont à l'endroit plat délicat 
        dict_in[key]["rolling_grad"][0:wdw_size] = dict_in[key]["rolling_grad"][wdw_size+1] 
                
    return 

for key in thigh_data:
    add_features_rolling_mean_wdw(thigh_data[key], 15)
    add_features_rolling_grad_wdw(thigh_data[key], 8)

#%%

def add_features_current_sign(dict_in):
    for key in dict_in:
        dict_in[key]["current_sign"] = (dict_in[key].current_sent==0).astype(int)
        
    return 

for key in thigh_data:
    add_features_current_sign(thigh_data[key])

#%%
# ess = (thigh_data["sub3"][5]["current_sign"]).to_numpy()
# a = (ess[:-1] != ess[1:]).astype(int)
# b = np.append(a,0)

def add_features_current_change(dict_in):
    for key in dict_in:
        temp = (dict_in[key]["current_sign"]).to_numpy()
        temp2 = (temp[:-1] != temp[1:]).astype(int)
        dict_in[key]["current_change"] = np.append(temp2,0)        
    return 

for key in thigh_data:
    add_features_current_change(thigh_data[key])
    
#%% static 


def add_stat_data(dict_in, counter):
    # pris de er_ut_d# de static_data.py
    static = [0.764743,1.393811,1.413101,0.563231,0.862935] 
    for key in dict_in:
        # l = len(dict_in[key]["t"])
        # u = np.full(shape = l, fill_value = static[counter])
        dict_in[key]["static_val"] = static[counter] #dict_in[key].apply(lambda x: u, axis = 1)
    return
    
c = 0
for key in thigh_data:
    add_stat_data(thigh_data[key], c)
    c = c + 1
#%%

save_obj(thigh_data, "thigh_good_gc_long_features.pkl")

#%%


def plot_ind_thigh(d1):
    idx = range( len(d1["t"]))
    
    plt.figure()
    plt.plot(d1["t"][idx], d1["current_sign"][idx], label = "current sign")
    # plt.plot(d1["t"][idx], d1["no_mc_thigh_angle"][idx], label = "body part")
    # plt.plot(d1["t"][idx], d1["res_norm_thigh"][idx], label = "res thigh")
    # plt.plot(d1["t"][idx], d1["res_norm_kmau"][idx], label = "res kmau")
    # plt.plot(d1["t"][idx], d1["force"][idx]/10, label = "force/10")
    # plt.plot(d1["t"][idx], d1["R_leg"][idx]*20, label = "R_leg")
    # plt.plot(d1["t"][idx], d1["L_leg"][idx]*20, label = "L_leg")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.legend()
    return 

# plot_ind_thigh(thigh_data["sub4"][33])

#%%


