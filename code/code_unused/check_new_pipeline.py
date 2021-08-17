# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:59:03 2021

@author: ksdiv

juste pr verif 
"""


import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

file_new = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\TEST2_Postprocessed_Mocap_angles.pkl"
file_old = r"E:\ETHZ\mast_sem_IV\pdm\code\mocap_angles\S01_Postprocessed_Mocap_angles_force_residuals.pkl"

data_new = op_pickle(file_new)
data_old = op_pickle(file_old)

mm = mean_squared_error(data_new["mc_shank_angle"], data_old["mc_shank_angle"])
m_thigh_ang  = mean_squared_error(data_new["mc_thigh_angle"], data_old["mc_thigh_angle"])
m_kmal_ang  = mean_squared_error(data_new["mc_kmal_angle"], data_old["mc_kmal_angle"])
m_kmau_ang  = mean_squared_error(data_new["mc_kmau_angle"], data_old["mc_kmau_angle"])


m_force  = mean_squared_error(data_new["force"], data_old["force"])
m_vgrf  = mean_squared_error(data_new["vgrf"], data_old["vgrf"])
m_r_n_shank  = mean_squared_error(data_new["res_norm_shank"], data_old["res_norm_shank"])


# trop stricte 
# a = data_new["mc_shank_angle"][23]
# b = data_old["mc_shank_angle"][23]
# u = type(data_new["mc_shank_angle"][23])

c_s_ang  = data_new["mc_shank_angle"].compare(data_old["mc_shank_angle"])

# assert pd.testing.assert_series_equal(data_new["mc_shank_angle"], data_old["mc_shank_angle"], check_dtype=False, check_index_type='equiv', check_series_type=False,  check_names=False, check_exact=False, check_datetimelike_compat=False, check_categorical=False, check_category_order=False, check_freq=False, check_flags=False, rtol=1e-05, atol=1e-05, obj='Series')
