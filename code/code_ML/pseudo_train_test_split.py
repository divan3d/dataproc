# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:15:11 2021

@author: ksdiv

prepares train/ test dataset for sklearn - 
changes from dict of dataframe to numpy 
randomizes order of the gait cycle through all subjects (specify seed)
(keeps one gait cycle together)

fait un pseudo - train_test_split 
"""

import pickle
import numpy as np 
from random import shuffle, seed 

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
# thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long_features.pkl"
# thigh_data = op_pickle(thigh_data_file)

# #%%

# # tjr mettre Y en premier - le reste features X 
# XY_names = ["no_mc_thigh_angle", "no_mc_kmau_angle", "current_sent", "rolling_mean", "rolling_grad"]
# # faut que [0,1]    
# test_size = 0.3 

#%%

def get_useful_val_dict(dict_in, dict_val, XY_names):
    counter = len(dict_val)
    for keys in dict_in:
        dict_val[counter] = dict_in[keys][XY_names]
        counter = counter + 1
    return 

def dict_to_list(dict_dict_in, XY_names):
    dict_val = {}
    # if len(dict_dict_in)==55 :
    #     get_useful_val_dict(dict_dict_in, dict_val, XY_names)
    #     return dict_val
    for keys in dict_dict_in:
        get_useful_val_dict(dict_dict_in[keys], dict_val, XY_names)
    return dict_val    
 
#%%   
def get_train_XY(XY_names, train):
    X_train = np.zeros((1,(len(XY_names)-1)))
    y_train = np.array([0])
    for idx in range(len(train)):
        y_train = np.append(y_train, train[idx][XY_names[0]].to_numpy())
        xtemp = train[idx][XY_names[1]].to_numpy()
        for idx_names in range(2,len(XY_names)):
            xtemp = np.column_stack((xtemp, train[idx][XY_names[idx_names]].to_numpy()))
        X_train = np.append(X_train,xtemp, axis = 0)
    X_train = X_train[1:,:]
    y_train = y_train[1:]
    return X_train, y_train
    

def get_test_XY(XY_names, test):
    X_test = np.zeros((1,(len(XY_names)-1)))
    y_test = np.array([0])
    for idx in range(len(test)):
        y_test = np.append(y_test, test[idx][XY_names[0]].to_numpy())
        xtemp = test[idx][XY_names[1]].to_numpy()
        for idx_names in range(2,len(XY_names)):
            xtemp = np.column_stack((xtemp, test[idx][XY_names[idx_names]].to_numpy()))
        X_test = np.append(X_test,xtemp, axis = 0)
    X_test = X_test[1:,:]
    y_test = y_test[1:]
    return X_test, y_test

def get_valid_XY(XY_names, valid):
    X_valid = np.zeros((1,(len(XY_names)-1)))
    y_valid = np.array([0])
    for idx in range(len(valid)):
        y_valid = np.append(y_valid, valid[idx][XY_names[0]].to_numpy())
        xtemp = valid[idx][XY_names[1]].to_numpy()
        for idx_names in range(2,len(XY_names)):
            xtemp = np.column_stack((xtemp, valid[idx][XY_names[idx_names]].to_numpy()))
        X_valid = np.append(X_valid,xtemp, axis = 0)
    X_valid = X_valid[1:,:]
    y_valid = y_valid[1:]
    return X_valid, y_valid


def train_test_split_pseudo(dict_dict_in, XY_names, test_size, rdm_seed = None):
    """
    extracts to numpy and separates values (writen in XY_names) into test/ train
    of dict which contains for each subject dict of all gait cycles 

    Parameters
    ----------
    dict_dict_in : dict : dict of dict, all (usable) gait cycles for one subject
                        is put into a dict, and all these dict are put into one
                        dict
    XY_names : list : names of the columns that want to extract 
                        !! 1st value : y column, the rest : x (features)
    test_size : [0, 1] : proportion of gait cycle used to test
    rdm_seed : int, optional : seed for random shuffling of gait cycles. The default is None.

    Returns
    -------
    X_train : array : contains (1- test_size) of gait cycle features
    y_train : array : contains (1- test_size) of gait cycle output values
    X_test : array : contains (test_size) of gait cycle features
    y_test : array : contains (test_size) of gait cycle output values

    """
    
    # get columns specified in XY_names form pd dataframe
    l_values = dict_to_list(dict_dict_in, XY_names)
    
    # randomize order of gait cycles 
    seed(a = rdm_seed )
    rdm_keys = list(l_values.keys())  
    shuffle(rdm_keys)    
    nbr_gait = len(rdm_keys)
    L = []
    for i in rdm_keys:
        L.append(l_values[i])
            
    # separate gait cycles into test/train set     
    cut_val = int(np.floor((1-test_size)*nbr_gait))
        
    train = L[:cut_val]
    test = L[cut_val:]
    
    X_train, y_train = get_train_XY(XY_names, train)
    X_test, y_test = get_test_XY(XY_names, test)
    
    return X_train, y_train, X_test, y_test

def train_test_valid_split_pseudo(dict_dict_in, XY_names, test_size, rdm_seed = None):
    """
    extracts to numpy and separates values (writen in XY_names) into test/ train / valid
    of dict which contains for each subject dict of all gait cycles 

    Parameters
    ----------
    dict_dict_in : dict : dict of dict, all (usable) gait cycles for one subject
                        is put into a dict, and all these dict are put into one
                        dict
    XY_names : list : names of the columns that want to extract 
                        !! 1st value : y column, the rest : x (features)
    test_size : [0, 1] : proportion of gait cycle used to test, divise par 2, l'autre est
                        utilisé pr validation set
    rdm_seed : int, optional : seed for random shuffling of gait cycles. The default is None.

    Returns
    -------
    X_train : array : contains (1- test_size) of gait cycle features
    y_train : array : contains (1- test_size) of gait cycle output values
    X_test : array : contains (test_size/2) of gait cycle features
    y_test : array : contains (test_size/2) of gait cycle output values
    X_valid : array : contains (test_size/2) of gait cycle features
    y_valid : array : contains (test_size/2) of gait cycle output values

    """
    
    # get columns specified in XY_names form pd dataframe
    l_values = dict_to_list(dict_dict_in, XY_names)
    
    # randomize order of gait cycles 
    seed(a = rdm_seed )
    rdm_keys = list(l_values.keys())  
    shuffle(rdm_keys)    
    nbr_gait = len(rdm_keys)
    L = []
    for i in rdm_keys:
        L.append(l_values[i])
            
    # separate gait cycles into test/train set     
    cut_val = int(np.floor((1-test_size)*nbr_gait))
    cut_2_val = int(np.floor((nbr_gait - cut_val)/2))
        
    train = L[:cut_val]
    test = L[cut_val:(cut_val+cut_2_val)]
    valid = L[(cut_val+cut_2_val):]
    
    X_train, y_train = get_train_XY(XY_names, train)
    X_test, y_test = get_test_XY(XY_names, test)
    X_valid, y_valid = get_valid_XY(XY_names, valid)
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid

# X_tr, y_tr, X_te, y_te = train_test_split_pseudo(thigh_data, XY_names, test_size)

#%%

def train_test_valid_split_pseudo_2(dict_in, XY_names, test_size, rdm_seed = None):
    """
    extracts to numpy and separates values (writen in XY_names) into test/ train / valid
    of dict which contains for each subject dict of all gait cycles 

    Parameters
    ----------
    dict_in : dict : contains dataframes, which are the gait cycles
    XY_names : list : names of the columns that want to extract 
                        !! 1st value : y column, the rest : x (features)
    test_size : [0, 1] : proportion of gait cycle used to test, divise par 2, l'autre est
                        utilisé pr validation set
    rdm_seed : int, optional : seed for random shuffling of gait cycles. The default is None.

    Returns
    -------
    X_train : array : contains (1- test_size) of gait cycle features
    y_train : array : contains (1- test_size) of gait cycle output values
    X_test : array : contains (test_size/2) of gait cycle features
    y_test : array : contains (test_size/2) of gait cycle output values
    X_valid : array : contains (test_size/2) of gait cycle features
    y_valid : array : contains (test_size/2) of gait cycle output values

    """
        
    # dict_in has keys0 to len of thingy 
    l_values = [range(0,len(dict_in))]
    
    # randomize order of gait cycles 
    seed(a = rdm_seed )
    rdm_keys = list(l_values.keys())  
    shuffle(rdm_keys)    
    nbr_gait = len(rdm_keys)
    L = []
    for i in rdm_keys:
        L.append(l_values[i])
            
    # separate gait cycles into test/train set     
    cut_val = int(np.floor((1-test_size)*nbr_gait))
    cut_2_val = int(np.floor((nbr_gait - cut_val)/2))
        
    train = L[:cut_val]
    test = L[cut_val:(cut_val+cut_2_val)]
    valid = L[(cut_val+cut_2_val):]
    
    X_train, y_train = get_train_XY(XY_names, train)
    X_test, y_test = get_test_XY(XY_names, test)
    X_valid, y_valid = get_valid_XY(XY_names, valid)
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid


