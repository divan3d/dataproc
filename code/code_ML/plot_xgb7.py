# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:39:22 2021

@author: ksdiv

plot results
"""

import matplotlib.pyplot as plt
import pickle
import statistics

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_max_depth_min_child_weight_3.pkl"
dico_results = op_pickle(f)

#%%

colors = ["b", "g", "r", "c", "m", "k"]

#%% plot results 
count = 0
plt.figure()
for sub in dico_results:
    score_val = []
    score_train  = []
    depth_weight = []
    
    for val in dico_results[sub]:
        score_val.append(dico_results[sub][val]["score validation"])
        score_train.append(dico_results[sub][val]["score train"])
        depth_weight.append([dico_results[sub][val]["max_depth"],dico_results[sub][val]["min_child_weight"]])
        
    

    plt.plot(score_val, "o-", label = "score validation set, separate subject " + sub, color = colors[count])
    plt.plot(score_train, "x-", label = "score on train set", color = colors[count])
    plt.legend()
    plt.title("effects of max depth and min child weight thigh")
    plt.xticks(range(9), depth_weight, rotation = 90)
    count = count + 1
    
#%%

l31 = []
l33 = []
l35 = []

l51 = []
l53 = []
l55 = []

l71 = []
l73 = []
l75 = []

l91 = []
l93 = []
l95 = []


for sub in dico_results:
    
    l31.append(dico_results[sub][0]["score validation"])
    l33.append(dico_results[sub][1]["score validation"])
    l35.append(dico_results[sub][2]["score validation"])
    
    l51.append(dico_results[sub][3]["score validation"])
    l53.append(dico_results[sub][4]["score validation"])
    l55.append(dico_results[sub][5]["score validation"])
    
    l71.append(dico_results[sub][6]["score validation"])
    l73.append(dico_results[sub][7]["score validation"])
    l75.append(dico_results[sub][8]["score validation"])
    
    # l91.append(dico_results[sub][9]["score validation"])
    # l93.append(dico_results[sub][10]["score validation"])
    # l95.append(dico_results[sub][11]["score validation"])


l31m = statistics.mean(l31)
l33m = statistics.mean(l33)
l35m = statistics.mean(l35)

l51m = statistics.mean(l51)
l53m = statistics.mean(l53)
l55m = statistics.mean(l55)

l71m = statistics.mean(l71)
l73m = statistics.mean(l73)
l75m = statistics.mean(l75)

# l91m = statistics.mean(l91)
# l93m = statistics.mean(l93)
# l95m = statistics.mean(l95)

plt.figure()
plt.plot([l31m, l33m, l35m, l51m, l53m, l55m, l71m, l73m, l75m], "o-")
# plt.plot([l31m, l33m, l35m, l51m, l53m, l55m, l71m, l73m, l75m, l91m, l93m, l95m], "o-")
plt.xticks(range(9), depth_weight, rotation = 90)
    
#%% look at features

list_keys = list(X_train)
#%%
count = 0
plt.figure()
for sub in dico_results:
    imp_feat = []
    for val in dico_results[sub]:
        imp_feat.append(dico_results[sub][val]["important features"])
        
   
    plt.plot(imp_feat[0])
    for v in range(len(imp_feat)):
        plt.plot(imp_feat[v], "o-", label = depth_weight[v], color = colors[count])
    plt.legend()
    plt.xticks(range(70), list_keys, rotation = 90)
    plt.title("important features change when max depth and min child weight vary 2nd exp")
    count = count + 1 


#%%

count = 0
plt.figure()
for sub in dico_results:
    tot_progress =[]
    for val in dico_results[sub]:
        tot_progress.append(dico_results[sub][val]["progress"])
    
    

    plt.grid()
    for v in range(len(tot_progress)):
        plt.plot(tot_progress[v]['validation_0']["rmse"], label='train '+ sub, color = colors[count])
        plt.plot(tot_progress[v]['validation_1']["rmse"], label='valid '+ sub, color = colors[count])
    plt.legend()
    count = count + 1
