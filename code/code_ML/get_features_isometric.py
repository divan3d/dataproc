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
        


#%%

f12 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_EQ_iso/SH_FL1_cut_Isometric2.pkl"
d12 = op_pickle(f12)

#%% dict of features

# input : iso experiment, which has 2 compressions
# extract these values and save as variable so don't have to retrieve each time

number = f12[-5]
off1 = d12["iso" + number + "_off_1"]
off2 = d12["iso" + number + "_off_2"]
on1 = d12["iso" + number + "_on_1"]
on2 = d12["iso" + number + "_on_2"]
dict_features = {}

def grad_d(series_in):
    """
    gets gradient 
    """ 
    grad = np.gradient(series_in)
    max_grad = max(grad)
    idx_max_grad = np.argmax(grad)
    min_grad = min(grad)
    idx_min_grad = np.argmin(grad)
    return max_grad, idx_max_grad, min_grad, idx_min_grad


def get_iso_on_enc_max_min_diff(on_1, on_2):
    enc_on1 = - on_1["HallSensor"]
    enc_on2 = - on_2["HallSensor"]
    max_enc_1 = enc_on1.max()
    max_enc_2 = enc_on2.max()
    max_val = (max_enc_1 + max_enc_2)/2
    
    min_enc_1 = enc_on1.min()
    min_enc_2 = enc_on2.min()
    min_val = (min_enc_1 + min_enc_2)/2
    
    diff1 = max_enc_1 - min_enc_1
    diff2 = max_enc_2 - min_enc_2
    diff_val = (diff1 + diff2)/2
    
    dict_features["iso_on_enc_max"] = max_val
    dict_features["iso_on_enc_min"] = min_val
    dict_features["iso_on_diff_min_max"] = diff_val
    return

def get_iso_off_enc_max_min_diff(off_1, off_2):
    enc_off1 = - off_1["HallSensor"]
    enc_off2 = - off_2["HallSensor"]
    max_enc_1 = enc_off1.max()
    max_enc_2 = enc_off2.max()
    max_val = (max_enc_1 + max_enc_2)/2
    
    min_enc_1 = enc_off1.min()
    min_enc_2 = enc_off2.min()
    min_val = (min_enc_1 + min_enc_2)/2
    
    diff1 = max_enc_1 - min_enc_1
    diff2 = max_enc_2 - min_enc_2
    diff_val = (diff1 + diff2)/2
    
    dict_features["iso_off_enc_max"] = max_val
    dict_features["iso_off_enc_min"] = min_val
    dict_features["iso_off_diff_min_max"] = diff_val
    return

def get_iso_on_enc_deriv(on_1, on_2):
    g_max1, g_max_idx1, g_min1, g_min_idx1 = grad_d(-on_1["HallSensor"])
    g_max2, g_max_idx2, g_min2, g_min_idx2 = grad_d(-on_2["HallSensor"]) 
    
    max_val = (g_max1 + g_max2)/2
    max_idx_val = (g_max_idx1 + g_max_idx2)/2
    
    dict_features["iso_on_enc_max_deriv"] = max_val
    dict_features["iso_on_enc_idx_max_deriv"] = max_idx_val    
    return

def get_iso_off_enc_deriv(off_1, off_2):
    g_max1, g_max_idx1, g_min1, g_min_idx1 = grad_d(-off_1["HallSensor"])
    g_max2, g_max_idx2, g_min2, g_min_idx2 = grad_d(-off_2["HallSensor"]) 
    
    min_val = (g_min1 + g_min2)/2
    min_idx_val = (g_min_idx1 + g_min_idx2)/2
    
    dict_features["iso_off_enc_min_deriv"] = min_val
    dict_features["iso_off_enc_idx_min_deriv"] = min_idx_val    
    return

def get_iso_on_gyro_c_s(on_1, on_2):
    gcs1 = on_1["GyroCShank"]
    gcs2 = on_2["GyroCShank"]
    max_gyroc_1 = gcs1.max()
    max_gyroc_2 = gcs2.max()
    max_val = (max_gyroc_1 + max_gyroc_2)/2
    
    min_gyroc_1 = gcs1.min()
    min_gyroc_2 = gcs2.min()
    min_val = (min_gyroc_1 + min_gyroc_2)/2
    
    idx_max_gc_1 = gcs1.idxmax()
    idx_max_gc_2 = gcs2.idxmax()
    idx_max = (idx_max_gc_1 + idx_max_gc_2)/2
    
    idx_min_gc_1 = gcs1.idxmin()
    idx_min_gc_2 = gcs2.idxmin()
    idx_min = (idx_min_gc_1 + idx_min_gc_2)/2
    
    width_pos1 = sum(gcs1[:25]>=5)
    width_pos2 = sum(gcs2[:25]>=5)
    width_pos = (width_pos1 + width_pos2)/2
    
    dict_features["iso_on_gyro_c_s_max"] = max_val
    dict_features["iso_on_gyro_c_s_min"] = min_val
    dict_features["iso_on_gyro_c_s_idx_max"] = idx_max
    dict_features["iso_on_gyro_c_s_idx_min"] = idx_min
    dict_features["iso_on_gyro_c_s_width_pos"] = width_pos
    return

def get_iso_on_gyro_c_t(on_1, on_2):
    gct1 = on_1["GyroCThigh"]
    gct2 = on_2["GyroCThigh"]
    max_gyroc_1 = gct1.max()
    max_gyroc_2 = gct2.max()
    max_val = (max_gyroc_1 + max_gyroc_2)/2
    
    min_gyroc_1 = gct1.min()
    min_gyroc_2 = gct2.min()
    min_val = (min_gyroc_1 + min_gyroc_2)/2
    
    idx_max_gc_1 = gct1.idxmax()
    idx_max_gc_2 = gct2.idxmax()
    idx_max = (idx_max_gc_1 + idx_max_gc_2)/2
    
    idx_min_gc_1 = gct1.idxmin()
    idx_min_gc_2 = gct2.idxmin()
    idx_min = (idx_min_gc_1 + idx_min_gc_2)/2
    
    width_pos1 = sum(gct1[:25]<=-5)
    width_pos2 = sum(gct2[:25]<=-5)
    width_pos = (width_pos1 + width_pos2)/2
    
    dict_features["iso_on_gyro_c_t_max"] = max_val
    dict_features["iso_on_gyro_c_t_min"] = min_val
    dict_features["iso_on_gyro_c_t_idx_max"] = idx_max
    dict_features["iso_on_gyro_c_t_idx_min"] = idx_min
    dict_features["iso_on_gyro_c_t_width_pos"] = width_pos
    return

def get_iso_off_gyro_c_s(off_1, off_2):
    min_g_1 = off_1["GyroCShank"].min()
    min_g_2 = off_2["GyroCShank"].min()
    min_g = (min_g_1 + min_g_2)/2
    
    imin_g_1 = off_1["GyroCShank"].idxmin()
    imin_g_2 = off_2["GyroCShank"].idxmin()
    imin_g = (imin_g_1 + imin_g_2)/2
    
    max_g_1 = off_1["GyroCThigh"].max()
    max_g_2 = off_2["GyroCThigh"].max()
    max_g = (max_g_1 + max_g_2)/2
    
    imax_g_1 = off_1["GyroCThigh"].idxmax()
    imax_g_2 = off_2["GyroCThigh"].idxmax()
    imax_g = (imax_g_1 + imax_g_2)/2
    
    dict_features["iso_off_gyro_c_s_min"] = min_g
    dict_features["iso_off_gyro_c_s_idx_min"] = imin_g
    dict_features["iso_off_gyro_c_t_max"] = max_g
    dict_features["iso_off_gyro_c_s_idx_max"] = imax_g
    return

#%%

get_iso_off_enc_deriv(off1, off2)
get_iso_on_enc_deriv(on1, on2)
get_iso_off_enc_max_min_diff(off1, off2)
get_iso_on_enc_max_min_diff(on1, on2)
get_iso_on_gyro_c_s(on1, on2)
get_iso_on_gyro_c_t(on1, on2)