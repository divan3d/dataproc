# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:46:04 2021

@author: ksdiv
"""

import pickle
import os
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
        
        
t1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\SA\INTERMEDIATE\SA_FUN_FL5_2.pkl"
d1 = op_pickle(t1)

# t3 = r"E:\ETHZ\mast_sem_IV\pdm\code02\SA_200721_FUN_FL3.pkl"
# d3 = op_pickle(t3)

# t5 = r"E:\ETHZ\mast_sem_IV\pdm\code02\SA_200721_FUN_FL5.pkl"
# d5 = op_pickle(t5)


#%%

def plt_shank(d_in, name):
    plt.figure()
    plt.plot(d_in["t"], d_in["no_mc_shank_angle"], label = "mocap shank")
    plt.plot(d_in["t"], d_in["no_mc_kmal_angle"], label = "mocap kmal")
    # plt.plot(d_in["t"], d_in["AlphaShank"], label = "imu")
    plt.legend()
    plt.title(name)
    return

def plt_thigh(d_in, name):
    plt.figure()
    plt.plot(d_in["t"], d_in["no_mc_thigh_angle"], label = "mocap thigh")
    plt.plot(d_in["t"], d_in["no_mc_kmau_angle"], label = "mocap kmau")
    # plt.plot(d_in["t"], d_in["AlphaShank"], label = "imu")
    plt.legend()
    plt.title(name)
    return
    
plt_shank(d1, "FL 1")
# plt_shank(d3, "FL 3")
# plt_shank(d5, "FL 5")
    
# plt_thigh(d3, "FL 3")
# plt_thigh(d5, "FL 5")
    

#%% plot FL together 

d1_s = d1["no_mc_shank_angle"][20602:20709]
d1_t = d1["no_mc_kmal_angle"][20602:20709]
d3_s = d3["no_mc_shank_angle"][34500:34618]
d3_t = d3["no_mc_kmal_angle"][34500:34618]
d5_s = d5["no_mc_shank_angle"][24009:24124]
d5_t = d5["no_mc_kmal_angle"][24009:24124]

d1_s = d1_s.reset_index()
d3_s = d3_s.reset_index()
d5_s = d5_s.reset_index()
d1_t = d1_t.reset_index()
d3_t = d3_t.reset_index()
d5_t = d5_t.reset_index()

d1_s = d1_s.drop(columns = ["index"])
d3_s = d3_s.drop(columns = ["index"])
d5_s = d5_s.drop(columns = ["index"])
d1_t = d1_t.drop(columns = ["index"])
d3_t = d3_t.drop(columns = ["index"])
d5_t = d5_t.drop(columns = ["index"])

# plt.figure()
# plt.plot(d1_s, label = "FL1 shank")    
# plt.plot(d1_t, label = "FL1 kmal")  
# plt.plot(d3_s, label = "FL3 shank")    
# plt.plot(d3_t, label = "FL3 kmal")  
# plt.plot(d5_s, label = "FL3 shank")    
# plt.plot(d5_t, label = "FL3 kmal") 

#%% 

# fig, axs = plt.subplots(1,3, sharey= True)
# axs[0].plot(d1["t"], d1["no_mc_shank_angle"], label = "mocap shank", color = "cornflowerblue")
# axs[0].plot(d1["t"], d1["no_mc_kmal_angle"], label = "mocap kmal", color = "lime")
# axs[0].legend()
# axs[0].set_title("Force Level 1")
# axs[0].set_xlabel("time [s]")
# axs[0].set_ylabel("angles [deg]")
# axs[1].plot(d3["t"], d3["no_mc_shank_angle"], label = "mocap shank", color = "cornflowerblue")
# axs[1].plot(d3["t"], d3["no_mc_kmal_angle"], label = "mocap kmal", color = "lime")
# axs[1].set_title("Force Level 3")
# axs[1].legend()
# axs[1].set_xlabel("time [s]")
# axs[2].plot(d5["t"], d5["no_mc_shank_angle"], label = "mocap shank", color = "cornflowerblue")
# axs[2].plot(d5["t"], d5["no_mc_kmal_angle"], label = "mocap kmal", color = "lime")
# axs[2].set_title("Force Level 5")
# axs[2].set_xlabel("time [s]")
# axs[2].legend()

# fig.suptitle("Shank angle estimation at same treadmill speed (0.9m/s)", fontsize = 20)


#%%

# fig, axs = plt.subplots(1,2, sharey= True)
# axs[0].plot(d3["t"], d3["no_mc_shank_angle"], label = "mocap shank", color = "cornflowerblue")
# axs[0].plot(d3["t"], d3["no_mc_kmal_angle"], label = "mocap kmal", color = "lime")
# axs[0].legend()
# axs[0].set_title("Speed : 0.9 m/s")
# axs[0].set_xlabel("time [s]")
# axs[0].set_ylabel("angles [deg]")
# axs[1].plot(d3["t"], d3["no_mc_shank_angle"], label = "mocap shank", color = "cornflowerblue")
# axs[1].plot(d3["t"], d3["no_mc_kmal_angle"], label = "mocap kmal", color = "lime")
# axs[1].set_title("Speed : 1.3 m/s")
# axs[1].legend()
# axs[1].set_xlabel("time [s]")

# fig.suptitle("Shank angle estimation at Force Level 3", fontsize = 20 ) 
    

#%%
# plt.figure()
# plt.plot(d1["t"], d1["no_mc_shank_angle"], label = "mocap shank", color = "cornflowerblue")
# plt.plot(d1["t"], d1["no_mc_kmal_angle"], label = "mocap kmal", color = "lime")
# plt.legend()
# plt.xlabel("time [s]")
# plt.ylabel("angles [deg]")
# plt.title("Shank angle, Mode : Transparent, Speed : 1.3 m/s")