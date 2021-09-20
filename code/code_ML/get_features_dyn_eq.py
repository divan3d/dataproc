# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 13:06:28 2021

@author: ksdiv
"""

import pickle
import numpy as np
import os

# peut juste ajouter les features directement ds une nvll colonne 
# du coup autant le faire en meme temps que les autres ds get_features

#%%
subject = "SA"
dir_name_sub = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject

#%% 

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

#%%



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
    min_val = min(d_in["current_sent"][:110])
    offset = 0.2
    lim_val = min_val + offset
    nbr_pts = sum(d_in["current_read"] <= lim_val)
    d_in["dyn_eq_current_read_min"] = min_val
    d_in["dyn_eq_current_read_width_neg"] = nbr_pts
    return

#%%

dir_eq_shank = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + subject + "//" + subject + "_EQ_GC_shank"
dir_eq_thigh = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + subject + "//" + subject + "_EQ_GC_thigh"

shank_file = os.listdir(dir_eq_shank)
thigh_file = os.listdit(dir_eq_thigh)

for shank_f in shank_file:
    eq_shank = op_pickle(dir_eq_shank + "//" + shank_f)
    for key in eq_shank:
        get_encoder_dyn()
        get_current_sent_dyn()




