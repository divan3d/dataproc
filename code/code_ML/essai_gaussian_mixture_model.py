# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:31:11 2021

@author: ksdiv
essai gaussian mixture model 
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats 
from random import shuffle
from sklearn.mixture import GaussianMixture 

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
thigh_data_file = r"E:\ETHZ\mast_sem_IV\pdm\code\thigh_good_gc_long.pkl"
thigh_data = op_pickle(thigh_data_file)


def plot_sep_thigh(ex_dict):
    """
    plots each gait cycle separetely to form whole data
    - to check that individual gaits were cut correctly 
    
    Parameters
    ----------
    ex_dict : dictionary of dataframe, each containing individual gait cycle

    Returns
    -------
    None.

    """
    plt.figure()
    for key in ex_dict:
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_thigh_angle"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["no_mc_kmau_angle"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["res_norm_thigh"], label = key)
        plt.plot(ex_dict[key]["t"], ex_dict[key]["res_norm_kmau"], label = key)
        plt.legend()
    return

# plot_sep_thigh(thigh_data["sub6"])
#%% GMM - fonctionne pas 
gm_t = thigh_data["sub6"][12]
x_gm_t = gm_t["no_mc_kmau_angle"].to_numpy()
x_idx = np.arange(len(x_gm_t))
x_es = np.column_stack((x_idx, x_gm_t))

gm = GaussianMixture(n_components=2).fit(x_es)
#%%
gm_means = gm.means_
gm_var = np.diag(gm.covariances_[0])
#%%
sig1 = m.sqrt(gm_var[0])

# b1 = np.linspace(gm_means[0,0] - 3 * sig1, gm_means[0,0] + 3* sig1, 100)
b1 = np.linspace(-100, 100, 10000)
sig2 = m.sqrt(gm_var[1])

b2 = np.linspace(gm_means[1,0] - 3 * sig2, gm_means[1,0] + 3* sig2, 100)

a = stats.norm.pdf(b1,gm_means[0,0],sig1)

plt.figure()
plt.plot(x_idx, x_gm_t)
plt.plot(b1, stats.norm.pdf(b1,gm_means[0,0],sig1))
plt.plot(b2, stats.norm.pdf(b2,gm_means[1,0],sig2))