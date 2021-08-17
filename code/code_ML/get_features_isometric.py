# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:21:48 2021

@author: ksdiv

get features from isometric exp
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
import os

#%% juste la pour tester fonction, mais apres pr ouvrir les fichiers, sera fait dans get_features

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
file1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_FL3_TEST\SA_FL3_TEST_Isometric1.pkl"
data1 = op_pickle(file1)
file2 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_FL3_TEST\SA_FL3_TEST_Isometric2.pkl"
data2 = op_pickle(file2)

filet = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\INTERMEDIATE\SA_200721_FUN_FL3_MS.pkl"
datat = op_pickle(filet)

#%%

plt.figure()
plt.plot(data1["no_mc_thigh_angle"])
plt.plot(data1["no_mc_kmau_angle"])
plt.plot(data1["current_sent"])

plt.figure()
plt.plot(data2["no_mc_thigh_angle"])
plt.plot(data2["no_mc_kmau_angle"])
plt.plot(data2["current_sent"])

plt.figure()
plt.plot(datat["no_mc_thigh_angle"])
plt.plot(datat["no_mc_kmau_angle"])
