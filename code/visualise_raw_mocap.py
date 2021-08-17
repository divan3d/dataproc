# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:59:49 2021

@author: ksdiv
juste pour visualiser un peu 
voulait voir si il y a des vibrations a enlever mais ça a pas 
l'air ?? 
"""

import os
from scipy import interpolate as itp
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def file_save(obj, name, save_file, pickle_file):
    #pickkle_file que peut etre True si c'est un dataframe 
    if pickle_file:
        obj.to_pickle(name)
    
    if save_file :
        os.chdir(dname)
        save_obj(obj, name )
    return
                
#%% 

tpathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\experiment\2021_07_09_BALGRIST\Kanako Cal 02.pkl"
mocap = op_pickle(tpathMoCap)

old_pathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MoCap_Logs\S01_19112020\S01_19112020_Trial02.pkl"
o_mocap = op_pickle(old_pathMoCap)

#%% plot pour voir si il y a des vibrations, surtout pr les points qui sont sur des pilones 

def plot_3_axes_one_marker(marker,t,name):
    fig, axes = plt.subplots(3, sharex = True)
    axes[0].plot(t, marker[:,0], label = "x")
    axes[0].legend()
    axes[1].plot(t, marker[:,1], label = "y")
    axes[1].legend()
    axes[2].plot(t, marker[:,2], label = "z")
    axes[2].legend()
    fig.suptitle(name)
    return

def plot_compare_x_axis(marker1,marker2,marker3,marker4,marker5,t):    
    fig, axes = plt.subplots(5, sharex = True)
    axes[0].plot(t, marker1[:,0], label = "x marker1")
    axes[0].legend()
    axes[1].plot(t, marker2[:,0], label = "x marker2")
    axes[1].legend()
    axes[2].plot(t, marker3[:,0], label = "x marker3")
    axes[2].legend()
    axes[3].plot(t, marker4[:,0], label = "x marker4")
    axes[3].legend()
    axes[4].plot(t, marker5[:,0], label = "x marker5")
    axes[4].legend()
    return

#%%
tn = mocap["time"]

rightshank1 = mocap["points_array"][:,7,:]
rightshank4 = mocap["points_array"][:,10,:]
rightshank2 = mocap["points_array"][:,8,:]
rightshank3 = mocap["points_array"][:,9,:]
rightshank5 = mocap["points_array"][:,11,:]

# plot_compare_x_axis(rightshank1, rightshank2, rightshank3, rightshank4, rightshank5, tn)

#%%

plot_3_axes_one_marker(rightshank1, tn, "RightShank1")
plot_3_axes_one_marker(rightshank2, tn, "RightShank2")
plot_3_axes_one_marker(rightshank3, tn, "RightShank3")
plot_3_axes_one_marker(rightshank4, tn, "RightShank4")
plot_3_axes_one_marker(rightshank5, tn, "RightShank5")

#%%

to = o_mocap["time"]
os1 = o_mocap["points_array"][:,0,:]
os2 = o_mocap["points_array"][:,1,:]
os3 = o_mocap["points_array"][:,2,:]

plot_3_axes_one_marker(os1,to, "old_shank1")
plot_3_axes_one_marker(os2,to, "old_shank2")
plot_3_axes_one_marker(os3,to, "old_shank3")

#%%

to = o_mocap["time"]
ot3 = o_mocap["points_array"][:,12,:]
ot4 = o_mocap["points_array"][:,13,:]
ot5 = o_mocap["points_array"][:,14,:]

plot_3_axes_one_marker(ot4,to, "old_thigh4")
# plot_3_axes_one_marker(ot2,to, "old_thigh2")
# plot_3_axes_one_marker(ot3,to, "old_thigh3")

#%%

# rightthigh1 = mocap["points_array"][:,16,:]
# rightthigh2 = mocap["points_array"][:,17,:]
rightthigh3 = mocap["points_array"][:,18,:]
# rightthigh4 = mocap["points_array"][:,19,:]

# plot_3_axes_one_marker(rightthigh1,tn,"RightThigh1")
# plot_3_axes_one_marker(rightthigh2,tn,"RightThigh2")
plot_3_axes_one_marker(rightthigh3,tn,"RightThigh3")
# plot_3_axes_one_marker(rightthigh4,tn,"RightThigh4")

# #%%
# to = o_mocap["time"]
# orightthigh1 = o_mocap["points_array"][:,11,:]
# plot_3_axes_one_marker(orightthigh1, to, "o thigh")

# #%%

# kmal1 = mocap["points_array"][:,12,:]
# kmal2 = mocap["points_array"][:,13,:]
# kmal3 = mocap["points_array"][:,14,:]
# kmal4 = mocap["points_array"][:,15,:]

# plot_3_axes_one_marker(kmal1, tn, "KMAL1")
# plot_3_axes_one_marker(kmal2, tn, "KMAL2")
# plot_3_axes_one_marker(kmal3, tn, "KMAL3")
# plot_3_axes_one_marker(kmal4, tn, "KMAL4")

# #%%

# kmau1 = mocap["points_array"][:,20,:]
# kmau2 = mocap["points_array"][:,21,:]
# kmau3 = mocap["points_array"][:,22,:]
# kmau4 = mocap["points_array"][:,23,:]

# plot_3_axes_one_marker(kmau1, tn, "KMAU1")
# plot_3_axes_one_marker(kmau2, tn, "KMAU2")
# plot_3_axes_one_marker(kmau3, tn, "KMAU3")
# plot_3_axes_one_marker(kmau4, tn, "KMAU4")

#%%
# to = o_mocap["time"]
# s1 = o_mocap["points_array"][:,0,:]
# plot_3_axes_one_marker(s1,to)