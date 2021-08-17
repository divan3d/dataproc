# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:41:16 2021
error calculation for mocap data for all subjects for thigh and shank 

@author: ksdiv
"""

import pickle
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import mean_squared_error

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
#%% get data
    
file1 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S01_Mocap_angles_removed_offset_res.pkl"
file3 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S03_Mocap_angles_removed_offset_res.pkl"    
file4 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S04_Mocap_angles_removed_offset_res.pkl"
file5 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S05_Mocap_angles_removed_offset_res.pkl"
file6 = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S06_Mocap_angles_removed_offset_res.pkl"

data1 = op_pickle(file1)
data3 = op_pickle(file3)
data4 = op_pickle(file4)
data5 = op_pickle(file5)
data6 = op_pickle(file6)
#%% get cut data 
# file1c = r"S01_cut_Mocap_angles"
# file3c = r"S03_cut_Mocap_angles"
# file4c = r"S04_cut_Mocap_angles"
# file5c = r"S05_cut_Mocap_angles"
# file6c = r"S06_cut_Mocap_angles"

# data1c = op_pickle(file1c)
# data3c = op_pickle(file3c)
# data4c = op_pickle(file4c)
# data5c = op_pickle(file5c)
# data6c = op_pickle(file6c)

#%% get MSE for thigh and shank 

def get_mse_thigh(data):
    m = mean_squared_error(data["mc_thigh_angle"], data["mc_kmau_angle"])
    # m = mean_squared_error(data["no_mc_thigh_angle"], data["no_mc_kmau_angle"])
    return m 

def get_mse_shank(data):
    ms = mean_squared_error(data["mc_shank_angle"], data["mc_kmal_angle"])
    # ms = mean_squared_error(data["no_mc_shank_angle"], data["no_mc_kmal_angle"])
    return ms 

#%%

# mse_thigh_1 = get_mse_thigh(data1)
# mse_shank_1 = get_mse_shank(data1)

# mse_thigh_3 = get_mse_thigh(data3)
# mse_shank_3 = get_mse_shank(data3)

# mse_thigh_4 = get_mse_thigh(data4)
# mse_shank_4 = get_mse_shank(data4)

# mse_thigh_5 = get_mse_thigh(data5)
# mse_shank_5 = get_mse_shank(data5)

# mse_thigh_6 = get_mse_thigh(data6)
# mse_shank_6 = get_mse_shank(data6)

# mean_shank = statistics.mean([mse_shank_1, mse_shank_3, mse_shank_4, mse_shank_5, mse_shank_6])
# mean_thigh = statistics.mean([mse_thigh_1, mse_thigh_3, mse_thigh_4, mse_thigh_5, mse_thigh_6])

#%% get median for residuals 

median_1_shank = statistics.median(data1["res_norm_shank"])
median_3_shank = statistics.median(data3["res_norm_shank"])
median_4_shank = statistics.median(data4["res_norm_shank"])
median_5_shank = statistics.median(data5["res_norm_shank"])
median_6_shank = statistics.median(data6["res_norm_shank"])

mean_med_shank = statistics.mean([median_1_shank, median_3_shank, median_4_shank, median_5_shank, median_6_shank])

#%%

median_1_thigh = statistics.median(data1["res_norm_thigh"])
median_3_thigh = statistics.median(data3["res_norm_thigh"])
median_4_thigh = statistics.median(data4["res_norm_thigh"])
median_5_thigh = statistics.median(data5["res_norm_thigh"])
median_6_thigh = statistics.median(data6["res_norm_thigh"])

mean_med_thigh = statistics.mean([median_1_thigh, median_3_thigh, median_4_thigh, median_5_thigh, median_6_thigh])

#%% 
median_1_kmau = statistics.median(data1["res_norm_kmau"])
median_3_kmau = statistics.median(data3["res_norm_kmau"])
median_4_kmau = statistics.median(data4["res_norm_kmau"])
median_5_kmau = statistics.median(data5["res_norm_kmau"])
median_6_kmau = statistics.median(data6["res_norm_kmau"])

mean_med_kmau = statistics.mean([median_1_kmau, median_3_kmau, median_4_kmau, median_5_kmau, median_6_kmau])

#%% 
median_1_kmal = statistics.median(data1["res_norm_kmal"])
median_3_kmal = statistics.median(data3["res_norm_kmal"])
median_4_kmal = statistics.median(data4["res_norm_kmal"])
median_5_kmal = statistics.median(data5["res_norm_kmal"])
median_6_kmal = statistics.median(data6["res_norm_kmal"])

mean_med_kmal = statistics.mean([median_1_kmal, median_3_kmal, median_4_kmal, median_5_kmal, median_6_kmal])

#%% get MSE for indiv gait cycle 

def get_mse_shank_ind(data):
    ind_val_shank = []
    for key in data:
        # mse = mean_squared_error(data[key]["mc_shank_angle"], data[key]["mc_kmal_angle"])
        mse = mean_squared_error(data[key]["no_mc_shank_angle"], data[key]["no_mc_kmal_angle"])
        ind_val_shank.append(mse)
        data[key]["mse_shank"] = mse
    return ind_val_shank

def get_mse_thigh_ind(data):
    ind_val_thigh = []
    for key in data:
        # mse = mean_squared_error(data[key]["mc_thigh_angle"], data[key]["mc_kmau_angle"])
        mse = mean_squared_error(data[key]["no_mc_thigh_angle"], data[key]["no_mc_kmau_angle"])
        data[key]["mse_thigh"] = mse
        ind_val_thigh.append(mse)
    return ind_val_thigh

#%% prblm data5c

# data5c.pop(0)
# #%%
# ind_shank_d1 = get_mse_shank_ind(data1c)
# ind_thigh_d1 = get_mse_thigh_ind(data1c)
# ind_shank_d3 = get_mse_shank_ind(data3c)
# ind_thigh_d3 = get_mse_thigh_ind(data3c)
# ind_shank_d4 = get_mse_shank_ind(data4c)
# ind_thigh_d4 = get_mse_thigh_ind(data4c)
# ind_shank_d5 = get_mse_shank_ind(data5c)
# ind_thigh_d5 = get_mse_thigh_ind(data5c)
# ind_shank_d6 = get_mse_shank_ind(data6c)
# ind_thigh_d6 = get_mse_thigh_ind(data6c)

# #%% plot 

# fig, axe = plt.subplots()
# axe.boxplot([ind_shank_d1, ind_shank_d3, ind_shank_d4, ind_shank_d5, ind_shank_d6])
# axe.plot([1,2,3,4,5], [mean_shank, mean_shank, mean_shank, mean_shank, mean_shank], "bo", label = "MSE all subjects")
# axe.plot([1,2,3,4,5], [mse_shank_1,mse_shank_3,mse_shank_4,mse_shank_5,mse_shank_6], "ro", label = "MSE for each subject")
# axe.set_title("MSE separated gait cycle - Shank")
# axe.set_xlabel("subjects")
# axe.set_ylabel("MSE")
# axe.legend()

# #%%
# fig2, axe2 = plt.subplots()
# axe2.boxplot([ind_thigh_d1, ind_thigh_d3, ind_thigh_d4, ind_thigh_d5, ind_thigh_d6])
# axe2.plot([1,2,3,4,5], [mean_thigh, mean_thigh, mean_thigh, mean_thigh, mean_thigh], "bo", label = "MSE all subjects")
# axe2.plot([1,2,3,4,5], [mse_thigh_1,mse_thigh_3,mse_thigh_4,mse_thigh_5,mse_thigh_6], "ro", label = "MSE for each subject")
# axe2.set_title("MSE separated gait cycle - Thigh - with offset")
# axe2.set_xlabel("subjects")
# axe2.legend()