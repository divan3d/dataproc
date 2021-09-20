# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:59:41 2021

@author: ksdiv
"""

import pickle
import scipy.signal
import os
import glob

#%%
subject = "SH"
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

    
#%% get cadence, prend de 4000 a 5000 à chaque fois 

def indices_to_cut_R(d1):
    """
    finds indices to cut data such that left with ~ heel strike to toe off for 
    right leg -- corresponds approx. to heel strike right leg to heel strike left leg 

    Parameters
    ----------
    d1 : pd dataframe : dynamic data of 1 subject

    Returns
    -------
    peak_R_leg : array : contains indices of Right leg heel strike 

    """
    _, properties = scipy.signal.find_peaks(d1["R_leg"].values, height=1, plateau_size = 5)
    peak_R_leg = properties["left_edges"]
    return peak_R_leg

# take 1000 time stamps - approx. 10s and count nbr HS 
# dict_features = {}

def get_cadence(d_in):
    # might fuck up for SE FL5_2
    cadence = len(indices_to_cut_R(d_in[4000:5000]))/10
    dict_features["cadence"] = cadence
    return 

#%%

list_force = glob.glob(dir_name_sub + "\\" + subject + "_FL*")

for FL in list_force:
    fl  = FL[48:]
    list_con = glob.glob(dir_name_sub + "\\" + subject + "_FL" + fl + "\\" + subject + "_FL" + fl + "_Concentric*")
    for conc_file in list_con : 
        data_concentric = op_pickle(conc_file)
        new_folder = subject + "_features_dyn_uncut"
        dict_features = {}
        get_cadence(data_concentric)
        con = conc_file[-15:]
        save_in_new_file(new_folder, dict_features, "features_" + subject + "_FL" + fl + "_" + con)



