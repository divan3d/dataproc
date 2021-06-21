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
    
    #define path
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    #rename location variable
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
    
    location = 'PutLocationHere.csv'
    
    [Fprox,Fdist,t,Sync]=impcsv(location)
    
    data = [t,Fprox,Fdist,Sync]

    
    with open('Enterfilename','wb') as fp:
        pickle.dump(data,fp)
        

    
