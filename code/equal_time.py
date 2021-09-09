# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:18:41 2021

@author: ksdiv
get equal dist
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
from scipy import interpolate as itp
import pandas as pd
import os

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
f_sg_t_5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SG/SG_FL3/SG_FL3_Isometric1.pkl"
d_sg_t_5 = op_pickle(f_sg_t_5)

f_sb_t_1 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SB/SB_FL1/SB_FL1_Isometric2.pkl"
d_sb_t_1 = op_pickle(f_sb_t_1)

# f_sc_t_3 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SC/SC_CUT_EGC/thigh/SC_FL3_cut_thigh_1.pkl"
# d_sc_t_3 = op_pickle(f_sc_t_3)

f_sh_t_5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SH/SH_FL1/SH_FL1_Isometric2.pkl"
d_sh_t_5 = op_pickle(f_sh_t_5)

# f_sa_t_5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_CUT_EGC/thigh/SA_FL5_cut_thigh_1.pkl"
# d_sa_t_5 = op_pickle(f_sh_t_5)

# f_sa_s_5 = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/SA/SA_CUT_EGC/shank/SA_FL5_2_cut_shank_2.pkl"
# d_sa_s_5 = op_pickle(f_sa_s_5)

#%% get error by point 

def add_abs_err(dict_in):
    for key in dict_in:
        dict_in[key]["ae_thigh"] = abs(dict_in[key]["no_mc_thigh_angle"][:] - dict_in[key]["no_mc_kmau_angle"][:])
        dict_in[key]["ae_shank"] = abs(dict_in[key]["no_mc_shank_angle"][:] - dict_in[key]["no_mc_kmal_angle"][:])
    return

def get_equal_time(s_in, length):
    inter = itp.interp1d(range(len(s_in)) , s_in)
    out = inter(np.linspace(0, len(s_in)-1, length))
    return out 

def dict_equal_time(dict_in, length):
    eq_d1 = {}
    for key in dict_in:
        temp =  {}
        temp["no_mc_thigh_angle"] = get_equal_time(dict_in[key]["no_mc_thigh_angle"], length)
        temp["no_mc_kmau_angle"] = get_equal_time(dict_in[key]["no_mc_kmau_angle"], length)
        temp["no_mc_shank_angle"] = get_equal_time(dict_in[key]["no_mc_shank_angle"], length)
        temp["no_mc_kmal_angle"] = get_equal_time(dict_in[key]["no_mc_kmal_angle"], length)
        # temp["ae_thigh"] = get_equal_time(dict_in[key]["ae_thigh"], length)
        # temp["ae_shank"] = get_equal_time(dict_in[key]["ae_shank"], length)
        eq_d1[key] = pd.DataFrame.from_dict(temp)
    return eq_d1

def iso_equal_time(d_in, length):
    temp =  {}
    temp["no_mc_thigh_angle"] = get_equal_time(d_in["no_mc_thigh_angle"], length)
    temp["no_mc_kmau_angle"] = get_equal_time(d_in["no_mc_kmau_angle"], length)
    temp["no_mc_shank_angle"] = get_equal_time(d_in["no_mc_shank_angle"], length)
    temp["no_mc_kmal_angle"] = get_equal_time(d_in["no_mc_kmal_angle"], length)
    out_iso = pd.DataFrame.from_dict(temp)
    return out_iso

def get_mean(dict_in, name_key):
    res = list(dict_in.keys())[0]  # first key 
    temp_t = dict_in[res][name_key].to_numpy()
    for key in dict_in:
        temp_t = np.dstack((temp_t,dict_in[key][name_key].to_numpy()))
    mean_t = np.mean(temp_t, axis=2)
    return mean_t
    
def plot_sep(dict_in, name_key):
    plt.figure()
    plt.title("SG thigh and KMAU angles")
    for key in dict_in:
        plt.plot(dict_in[key][name_key], label = key)
        plt.plot(dict_in[key]["no_mc_kmau_angle"], label = key)
    return

def plot_iso(d_in):
    plt.figure()
    plt.plot(d_in["no_mc_kmau_angle"], label = "kmau")
    plt.plot(d_in["no_mc_thigh_angle"], label = "thigh")
    plt.plot(d_in["no_mc_kmal_angle"], label = "kmal")
    plt.plot(d_in["no_mc_shank_angle"], label = "shank")
    plt.legend()
    
def plot_f(d_in):
    plt.figure()
    plt.plot(d_in["no_mc_shank_angle"], label = "shank")
    plt.plot(d_in["no_mc_kmal_angle"], label = "kmal")
    plt.plot(d_in["no_mc_thigh_angle"], label = "thigh")
    plt.plot(d_in["no_mc_kmau_angle"], label = "kmau")
    plt.plot(d_in["Mode"])
    plt.plot(d_in["Force"])
    plt.legend()
    return

#%% iso

plot_f(d_sg_t_5)
plot_f(d_sb_t_1)
plot_f(d_sh_t_5)

eq_sg = iso_equal_time(d_sg_t_5, 170)
eq_sb = iso_equal_time(d_sb_t_1, 170)
eq_sh = iso_equal_time(d_sh_t_5, 170)


plot_iso(eq_sg)
plot_iso(eq_sb)
plot_iso(eq_sh)

#%%

a= np.shape(range(len(d_sg_t_5)))
b = np.shape(d_sg_t_5["no_mc_shank_angle"])

#%%
# add_abs_err(d_sg_t_5)
# add_abs_err(d_sb_t_1)
# add_abs_err(d_sc_t_3)
# add_abs_err(d_sh_t_5)

# eq_g_5 = dict_equal_time(d_sg_t_5, 150)
# eq_b_1 = dict_equal_time(d_sb_t_1, 150)
# eq_c_3 = dict_equal_time(d_sc_t_3, 150)
# eq_h_5 = dict_equal_time(d_sh_t_5, 150)

# mean_g_5_err_thigh = get_mean(eq_g_5, "ae_thigh")
# mean_b_1_err_thigh = get_mean(eq_b_1, "ae_thigh")
# mean_c_3_err_thigh = get_mean(eq_c_3, "ae_thigh")
# mean_h_5_err_thigh = get_mean(eq_h_5, "ae_thigh")

#%%
# plot_sep(eq_g_5, "no_mc_thigh_angle")
# plt.figure()
# plt.plot(mean_g_5_err_thigh[0,:])

# plot_sep(eq_b_1, "no_mc_thigh_angle")
# plt.figure()
# plt.plot(mean_b_1_err_thigh[0,:])

# plot_sep(eq_c_3, "no_mc_thigh_angle")
# plt.figure()
# plt.plot(mean_c_3_err_thigh[0,:])

# plot_sep(eq_h_5, "no_mc_thigh_angle")
# plt.figure()
# plt.plot(mean_h_5_err_thigh[0,:])

# #%%
# mean_sg_t = get_mean(eq_g_5, "no_mc_thigh_angle")
# mean_sg_ku = get_mean(eq_g_5, "no_mc_kmau_angle")

# plt.figure()
# plt.plot(mean_sg_t[0,:])
# plt.plot(mean_sg_ku[0,:])
# plt.plot(mean_g_5_err_thigh[0,:])
# plt.title("SG mean thigh and KMAU angles with error")

# #%%

# mean_thigh_g_5 = get_mean(eq_g_5, "no_mc_thigh_angle")
# mean_thigh_b_1 = get_mean(eq_b_1, "no_mc_thigh_angle")
# mean_thigh_c_3 = get_mean(eq_c_3, "no_mc_thigh_angle")
# mean_thigh_h_5 = get_mean(eq_h_5, "no_mc_thigh_angle")

# plt.figure()
# plt.plot(mean_thigh_g_5[0,:])
# plt.plot(mean_thigh_b_1[0,:])
# plt.plot(mean_thigh_c_3[0,:])
# plt.plot(mean_thigh_h_5[0,:])


# #%%


# file_name = r"E:/ETHZ/mast_sem_IV/pdm/extracted_data/" + "SA" + "/" + "SA" + "_CUT_EGC/shank"     
# list_files = os.listdir(file_name)
    

# temp_list_thigh = []
# for f_name in list_files:
#     file_now = op_pickle(file_name + "/" + f_name)
#     add_abs_err(file_now)
#     eq_time = dict_equal_time(file_now,150)
#     temp_list_thigh.append(get_mean(eq_time, "ae_shank"))


# #%%

# plt.figure()
# plt.plot(temp_list_thigh[0][0,:], label = "fl1,1")
# plt.plot(temp_list_thigh[1][0,:], label = "fl1,2")
# plt.plot(temp_list_thigh[2][0,:], label = "fl3,1")
# plt.plot(temp_list_thigh[3][0,:], label = "fl3,2")
# plt.plot(temp_list_thigh[4][0,:], label = "fl5,1")
# plt.plot(temp_list_thigh[5][0,:], label = "fl5,2")
# plt.legend()
# plt.title("shank error SA")

# #%%  
# # 24 ? -126

# fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_shank_angle"], label = "body")
# ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_kmal_angle"],label = "orthosis")
# # ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
# ax1.legend()
# ax1.set_ylabel("Shank angle [deg]", fontsize = 12)

# ax2.plot(np.linspace(0,100,129),d_sa_t_5[55]["no_mc_thigh_angle"],label = "body")
# ax2.plot(np.linspace(0,100,129),d_sa_t_5[55]["no_mc_kmau_angle"], label = "orthosis")
# ax2.set_ylabel("Thigh angle [deg]", fontsize = 12)
# ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
# ax2.legend()
# fig.suptitle("Leg angles related to gait cycle", fontsize = 20)

# #%%
# # 55 91
# ae_temp = abs(d_sa_s_5[296]["no_mc_shank_angle"]-d_sa_s_5[296]["no_mc_kmal_angle"])

# fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,97),d_sa_s_5[78]["no_mc_shank_angle"], label = "body")
# ax1.plot(np.linspace(0,100,97),d_sa_s_5[78]["no_mc_kmal_angle"],label = "orthosis")
# # ax1.plot(np.linspace(0,100,129),d_sa_s_5[55]["Force"])
# ax1.legend()
# ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
# ax2.plot(np.linspace(0,100,97),d_sa_s_5[296]["Force"])
# # ax2.plot(np.linspace(0,100,129),ae_temp)
# ax2.set_ylabel("abs error [-]", fontsize = 12)
# ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
# fig.suptitle("Shank angles & error related to gait cycle", fontsize = 12)

# #%%
# plt.figure()
# plt.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_shank_angle"], label = "body")
# plt.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_kmal_angle"],label = "orthosis")
# plt.xlabel("Percentage gait cycle [%]", fontsize = 14)
# plt.xlim((0,100))
# plt.ylim((-47,35))
# plt.ylabel("Shank angle [deg]", fontsize = 14)
# plt.legend()
# plt.title("Shank angle related to gait cycle", fontsize = 24)

# plt.figure()
# plt.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_shank_angle"], label = "body")
# plt.xlabel("Percentage gait cycle [%]", fontsize = 14)
# plt.ylabel("Shank angle [deg]", fontsize = 14)
# plt.xlim((0,100))
# plt.ylim((-47,35))
# # plt.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_kmal_angle"],label = "orthosis")
# plt.title("Shank angle related to gait cycle", fontsize = 24)
# #%%
# add_abs_err(d_sa_t_5)
# add_abs_err(d_sa_s_5)
# eq_a_s_5 = dict_equal_time(d_sa_s_5, 150)
# eq_a_t_5 = dict_equal_time(d_sa_t_5, 150)

# mean_sa_t = get_mean(eq_a_t_5, "no_mc_thigh_angle")
# mean_sa_ku = get_mean(eq_a_t_5, "no_mc_kmau_angle")
# mean_sa_s = get_mean(eq_a_s_5, "no_mc_shank_angle")
# mean_sa_kl = get_mean(eq_a_s_5, "no_mc_kmal_angle")
# mean_sa_et = get_mean(eq_a_t_5, "ae_thigh")
# mean_sa_es = get_mean(eq_a_s_5, "ae_shank")

# plt.figure()
# plt.plot(mean_sa_t[0,:])
# plt.plot(mean_sa_ku[0,:])
# plt.plot(mean_sa_et[0,:])

# #%%
# plt.figure()
# plt.plot(mean_sa_s[0,:])
# plt.plot(mean_sa_kl[0,:])
# plt.plot(mean_sa_es[0,:])

# #%% bricolage 

# temp_shank = d_sa_s_5[295].iloc[88:]
# temp_shank2 = d_sa_s_5[296].iloc[:88]
# fin = pd.concat([temp_shank,temp_shank2], ignore_index = True)

# ae_temp = abs(fin["no_mc_shank_angle"]-fin["no_mc_kmal_angle"])

# fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True, sharex = True)
# ax1.plot(np.linspace(0,100,96),fin["no_mc_shank_angle"], label = "body")
# ax1.plot(np.linspace(0,100,96),fin["no_mc_kmal_angle"],label = "orthosis")
# # ax1.plot(np.linspace(0,100,96),fin["Force"])
# ax1.legend()
# ax1.set_ylabel("Shank angle [deg]", fontsize = 12)
# ax1.grid(axis = "y")
# ax2.plot(np.linspace(0,100,96),ae_temp)
# # ax2.plot(np.linspace(0,100,129),ae_temp)
# ax2.set_ylabel("abs error [-]", fontsize = 12)
# ax2.grid(axis = "y")
# ax2.set_xlabel("Percentage gait cycle [%]", fontsize = 12)
# fig.suptitle("Shank angles & error related to gait cycle", fontsize = 12)

# #%%
# plt.figure()
# plt.plot(np.linspace(0,100,96),fin["no_mc_shank_angle"], label = "body")
# plt.plot(np.linspace(0,100,96),fin["no_mc_kmal_angle"],label = "orthosis")
# plt.xlabel("Percentage gait cycle [%]", fontsize = 14)
# plt.xlim((0,100))
# plt.grid(axis = "y")
# plt.ylim((-47,40))
# plt.ylabel("Shank angle [deg]", fontsize = 14)
# plt.legend()
# plt.title("Shank angle related to gait cycle", fontsize = 24)

# plt.figure()
# plt.plot(np.linspace(0,100,96),fin["no_mc_shank_angle"], label = "body")
# plt.xlabel("Percentage gait cycle [%]", fontsize = 14)
# plt.ylabel("Shank angle [deg]", fontsize = 14)
# plt.xlim((0,100))
# plt.ylim((-47,40))
# plt.grid(axis = "y")
# # plt.plot(np.linspace(0,100,129),d_sa_s_5[55]["no_mc_kmal_angle"],label = "orthosis")
# plt.title("Shank angle related to gait cycle", fontsize = 24)