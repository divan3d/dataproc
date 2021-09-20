# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:31:05 2021

@author: ksdiv
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/new_shank_gc.pkl"
d = op_pickle(f)

#%%

df = pd.DataFrame(data,columns=['A','B','C'])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()