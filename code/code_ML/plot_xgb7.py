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
    
f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/dict_max_depth_min_child_weight_shank_rm4.pkl"
dico_results2 = op_pickle(f)

#%%

colors = ["b", "g", "r", "c", "m", "k"]

#%% plot results 
def plot_val_and_train(d_results):
    count = 0
    plt.figure()
    for sub in d_results:
        score_val = []
        score_train  = []
        depth_weight = []
        
        for val in d_results[sub]:
            score_val.append(d_results[sub][val]["score validation"])
            score_train.append(d_results[sub][val]["score train"])
            depth_weight.append([d_results[sub][val]["max_depth"],d_results[sub][val]["reg_lambda"]])
            # depth_weight.append([d_results[sub][val]["col_sample"],d_results[sub][val]["subsample"]])
            
        
    
        plt.plot(score_val, "o-", label = "score validation set, separate subject " + sub, color = colors[count])
        plt.plot(score_train, "x-", label = "score on train set", color = colors[count])
        plt.legend()
        plt.title("effects of max depth and min child weight shank compliance")
        plt.xticks(range(12), depth_weight, rotation = 90)
        count = count + 1
    return depth_weight
    
#%%
depth_weight = plot_val_and_train(dico_results)
depth_weight2 = plot_val_and_train(dico_results2)
    
#%% plot results 
count = 0
plt.figure()
for sub in dico_results:
    score_val = []
    score_train  = []
    depth_reg = []
    
    for val in dico_results[sub]:
        score_val.append(dico_results[sub][val]["score validation"])
        score_train.append(dico_results[sub][val]["score train"])
        depth_reg.append([dico_results[sub][val]["max_depth"],dico_results[sub][val]["gamma"]])
        
    

    plt.plot(score_val, "o-", label = "score validation set, separate subject " + sub, color = colors[count])
    plt.plot(score_train, "x-", label = "score on train set", color = colors[count])
    plt.legend()
    plt.title("effects of max depth and reg shank compliance")
    plt.xticks(range(10), depth_reg, rotation = 90)
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
plt.title("shank compliance mean")

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
    
    # l71.append(dico_results[sub][6]["score validation"])
    # l73.append(dico_results[sub][7]["score validation"])
    # l75.append(dico_results[sub][8]["score validation"])
    
    # l91.append(dico_results[sub][9]["score validation"])
    # l93.append(dico_results[sub][10]["score validation"])
    # l95.append(dico_results[sub][11]["score validation"])


l31m = statistics.mean(l31)
l33m = statistics.mean(l33)
l35m = statistics.mean(l35)

l51m = statistics.mean(l51)
l53m = statistics.mean(l53)
l55m = statistics.mean(l55)

# l71m = statistics.mean(l71)
# l73m = statistics.mean(l73)
# l75m = statistics.mean(l75)

# l91m = statistics.mean(l91)
# l93m = statistics.mean(l93)
# l95m = statistics.mean(l95)

plt.figure()
plt.plot([l31m, l33m, l35m, l51m, l53m, l55m], "o-")
# plt.plot([l31m, l33m, l35m, l51m, l53m, l55m, l71m, l73m, l75m, l91m, l93m, l95m], "o-")
plt.xticks(range(6), depth_weight, rotation = 90)
plt.title("shank compliance mean")
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
        plt.plot(imp_feat[v], "o-", label = depth_reg[v], color = colors[count])
    plt.legend()
    # plt.xticks(range(70), list_keys, rotation = 90)
    plt.xticks(range(16), list_keys, rotation = 90)
    plt.title("important features change when max depth and min child weight vary shank compliance")
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
        # plt.plot(tot_progress[v]['validation_0']["rmse"], label='train '+ sub, color = colors[count])
        plt.plot(tot_progress[v]['validation_1']["rmse"], label='valid '+ sub, color = colors[count])
    plt.legend()
    plt.title("shank compliance")
    count = count + 1

#%%
count = 0
plt.figure()
for sub in dico_results:
    nbr_iter = []
    
    for val in dico_results[sub]:
        nbr_iter.append(dico_results[sub][val]["best n iter"])
        
    

    plt.plot(nbr_iter, "o-", label = "nbr of iterations, separate subject " + sub, color = colors[count])
    plt.legend()
    plt.title("nbr iterations")
    plt.xticks(range(12), depth_reg, rotation = 90)
    count = count + 1
    
#%%


def get_mean_val(d_results):
    headers = [[] for i in range(len(d_results["SA"]))]

    for sub in d_results:
        for x in range( len(d_results["SA"])):
            headers[x].append(d_results[sub][x]["score validation"])
            
    header_mean = []
    for n in range(len(headers)):
        header_mean.append(statistics.mean(headers[n]))
        
    plt.figure()
    plt.plot(header_mean, "o-")
    plt.xticks(range(len(d_results["SA"])), depth_weight, rotation = 90)
    return header_mean


h = get_mean_val(dico_results)
    
