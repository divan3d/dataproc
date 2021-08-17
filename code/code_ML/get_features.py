# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:48:31 2021

@author: ksdiv

get features from new experiment
for isometric : get dict of isometric features from get_features_isometric
adds these features to concentric

l'idee c'est que ça ouvre et envoie les trucs depuis ici, les fonctions seront stoquées
sur get_features_isometric par exemple
"""

import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.signal
import statistics
from scipy import stats
import os

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
file1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\SA_FL3_TEST"

listfile = os.listdir(file1)

# files in directory listed alphabetically :
    # concentric1
    # concentric2
    # isometric1
    # isometric2
    # transparent 

#%% isometric

iso1 = op_pickle(file1 + "\\" + listfile[2])
iso2 = op_pickle(file1 + "\\" + listfile[3])


# du coup en sortie pourrait avoir une ou deux ou 3 constantes a voir, a mettre 
# dans données concentriques 

#pourrait faire pr que ça mette ds dictionnaire, comme ça mets direct ds titre

iso1_features = {}

iso1_features["ex_i1_1"] = 23
iso1_features["ex_i1_2"] = 3

iso2_features = {}
iso2_features["ex_i2_1"] = 22
iso2_features["ex_i2_2"] = 111
iso2_features["ex_i2_3"] = 29

#%% concentric

con1 = op_pickle(file1 + "\\" + listfile[0])
con2 = op_pickle(file1 + "\\" + listfile[1])

#%% 

def add_iso_features_to_concentric(iso_features, concentric_df):
    for key in iso_features: 
        concentric_df[key] = iso_features[key]
    return

add_iso_features_to_concentric(iso1_features, con1)
add_iso_features_to_concentric(iso2_features, con2)