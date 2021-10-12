# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:29:36 2021

@author: ksdiv
plots pour exp av learning rate 
"""

import matplotlib.pyplot as plt
import pickle
import statistics

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
f = r"E:/ETHZ/mast_sem_IV/pdm/code/code_ML/shank_error_learning_rate_1.pkl"
dico_results = op_pickle(f)

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
        
        for x in d_results[sub]:
            score_val.append(d_results[sub][x]["score validation"])
            score_train.append(d_results[sub][x]["score train"])
            depth_weight.append(d_results[sub][x]["learning rate"])
                # depth_weight.append([d_results[sub][val]["col_sample"],d_results[sub][val]["subsample"]])
                
            
        
        plt.plot(score_val, "o-", label = "score validation set, separate subject " + sub, color = colors[count])
        plt.plot(score_train, "x-", label = "score on train set", color = colors[count])
        plt.legend()
        plt.title("effects of max depth and min child weight shank compliance")
        plt.xticks(range(4), depth_weight, rotation = 90)
        count = count + 1
    return 

plot_val_and_train(dico_results)

#%% get mean by learning rate


def get_mean_val(d_results):
    headers = [[] for i in range(len(d_results["SA"]))]

    for sub in d_results:
        for x in range( len(d_results["SA"])):
            headers[x].append(d_results[sub][x]["score validation"])
            
    header_mean = []
    for n in range(len(headers)):
        header_mean.append(statistics.mean(headers[n]))
 
    plt.figure()
    for x in range(len(d_results["SA"])):
        plt.plot(headers[x], "o-")
        plt.legend()
        plt.xticks(range(6),["SA", "SD", "SE", "SF", "SG", "SH"], rotation = 90)
    return header_mean

mean_header = get_mean_val(dico_results)
