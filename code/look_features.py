# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:41:43 2021

@author: ksdiv
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
import pandas as pd


def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

#%%
# f_in = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_FL3/SC_FL3_Concentric2.pkl"
f_in = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/INTERMEDIATE/SC_FUN_SM_FL3.pkl"
d_in = op_pickle(f_in)


gci32 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SG/SG_EQ_iso/SG_FL3_cut_Isometric2.pkl"
d_in = op_pickle(gci32)

#%%
# d_in = d_in.reset_index()

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)
ax1.plot(d_in["current_sent"], label = "current_sent", color = "lightcoral")
ax1.plot(d_in["current_read"], label = "current_read", color = "red")
# ax1.set_xlim([0,100])
# ax1.set_ylim([-45,40])
# ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("current [A]", fontsize = 12)
ax2.plot(- d_in["HallSensor"], label = "hall sensor", color = "red")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("Encoder [-]", fontsize = 12)
ax2.legend()
# fig.suptitle("Mean shank angles & error related to gait cycle, FL1", fontsize = 12)

ax3.plot(d_in["no_mc_shank_angle"], label = "shank", color = "lightsteelblue")
ax3.plot(d_in["no_mc_kmal_angle"], label = "kmal", color = "blue")
ax3.legend()

#%%

plt.figure()
plt.plot(d_in["HallSensor"]/100 +10, label = "HS")
plt.plot(d_in["current_sent"], label = "CS")

#%%

fig, (ax1,ax2, ax3) = plt.subplots(3,1, constrained_layout = True, sharex = True)
ax1.plot(d_in["Force"], label = "force", color = "lightcoral")
# ax1.set_xlim([0,100])
# ax1.set_ylim([-45,40])
# ax1.grid(axis = "y")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
ax1.legend()
ax1.set_ylabel("Force [N]", fontsize = 12)
ax2.plot(- d_in["HallSensor"], label = "hall sensor", color = "red")
# ax2.plot(np.linspace(0,100,129),ae_temp)
ax2.set_ylabel("Encoder [-]", fontsize = 12)
ax2.legend()
# fig.suptitle("Mean shank angles & error related to gait cycle, FL1", fontsize = 12)

ax3.plot(d_in["no_mc_shank_angle"], label = "shank", color = "lightsteelblue")
ax3.plot(d_in["no_mc_kmal_angle"], label = "kmal", color = "blue")
ax3.legend()