# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:31:05 2021

@author: ksdiv
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_gait_cycle_shank.pkl"
d = op_pickle(f)

#%%
ds = list(d.items())
data_train, data_valid = train_test_split(ds, test_size=0.3, random_state=42)
data_train = dict(data_train)
df = pd.concat(data_train, axis=0).reset_index(drop=True)

#%%


#%%
df = df.drop(columns = ["GyroAThigh", "GyroAShank"])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
# sn.heatmap(corrMatrix)
plt.show()

#%%

df2  = df.drop(columns = ["vgrf", "vgrf1", "vgrf2", "L_leg", "R_leg", "Mode", "GyroAThigh", "GyroAShank"])
list_df2 = list(df2)
corrMatrix2 = df2.corr()
sn.heatmap(corrMatrix2, annot= True)
plt.show()

#%%

df3 = df2.drop(columns= ["iso_off_enc_min_deriv", "iso_on_enc_max_deriv", "iso_off_enc_max", "iso_off_enc_min", "iso_off_diff_min_max", "iso_on_enc_max", "iso_on_enc_min", "iso_on_diff_min_max"
                         , "iso_on_gyro_c_t_min", "iso_on_gyro_c_t_width_pos", "iso_off_gyro_c_s_min", "iso_off_gyro_c_s_idx_min", "iso_off_gyro_c_s_idx_max", 
                         "iso_off_current_read_max_deriv", "iso_off_current_read_min_double_deriv", "dyn_eq_enc_init", "dyn_eq_enc_max"] )
list_df3 = list(df3)
corrMatrix3 = df3.corr()
sn.heatmap(corrMatrix3, annot= True)
plt.show()

#%%

df4 = df3.drop(columns = ["AlphaShank", "AlphaThigh", "GyroBThigh", "GyroBShank", "GyroCThigh", "GyroCShank", "AccelAThigh","AccelAShank", "AccelBThigh","AccelBShank", "AccelCThigh","AccelCShank"])
list_df4 = list(df4)
corrMatrix4 = df4.corr()
sn.heatmap(corrMatrix4, annot= True)
plt.show()

#%%

# import xlsxwriter 

# workbook = xlsxwriter.Workbook("E:/ETHZ/mast_sem_IV/pdm/code/feature_names.xlsx")
# worksheet = workbook.add_worksheet()
# worksheet.write_column("A1", list_keys)
# workbook.close()