# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:58:05 2019

@author: Gleb Koginov
"""
import numpy as np
import csv
from matplotlib import pyplot as plt
import os
import pickle

def impcsv(location):
    
    #rename variable
    file2Process = location
    
    #open csv file
    file = open(file2Process, 'r')
    
    #extract data
    reader = csv.reader(file)
    timeVals = list([])
    Fdist = list([])
    Fprox = list([])
    Sync = list([])
    for line in reader:
        if len(line) > 0:
            timeVals.append(np.float(line[0]))
            Fdist.append(np.float(line[1]))
            Fprox.append(np.float(line[2]))
            Sync.append(np.float(line[3]))
    
    Fdist = np.array(Fdist)
    Fprox = np.array(Fprox)
    timeVals = np.array(timeVals)
    Sync = np.array(Sync)
        
    return [Fprox,Fdist,timeVals,Sync]

if __name__ == "__main__": 
    
    name = 'Forcegenerationwiggle02.csv'
    
    [prox,dist,t,Sync]=impcsv(name)
    
    dist = dist.astype(int)
    prox = prox.astype(int)
    
    with open('lookup_FC_1','rb') as fileopen:
        lookupLC1=pickle.load(fileopen)
    with open('lookup_FC_2','rb') as fileopen:
        lookupLC2=pickle.load(fileopen)
    
    Fdist = np.zeros(len(prox))
    Fprox = np.zeros(len(prox))
    Fdist = lookupLC1[dist,1]
    Fprox = lookupLC2[prox,1]
    

    