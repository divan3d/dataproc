# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:15:21 2020

File to save the angles outputed by VICON in a pickle file

@author: Marc
"""
import numpy as np
import csv
import pickle

#location of csv file containing the vicon angles
location = r"C:\Users\wirth\Documents\WorkTemp\S06_23112020_KMAMotion\S06_23112020_T02_KMAangles.csv"

#name of the output file
fileout = 'S06_23112020_T02_ViconAnglesKMA.pkl'

#angletype: 'Leg'  / 'KMA'
angletype = 'KMA'

#open file
file = open(location, 'r')

#initialize csv reader
reader = csv.reader(file)

#initialize running parameters
start=0
stop =0


if angletype == 'Leg':
    #initialize empty lists
    ViconKneeAngle = list([])
    ViconHipAngle = list([])

    #go through csv file and read out the hip and knee angle
    for line in reader: 
        if len(line) > 0:
    
            if line[0]=='1':
                start=1
            
            if start == 1 and line[0]=='Segments':
                stop=1
                
            if start == 1 and stop == 0:            
                ViconKneeAngle.append(np.float(line[5]))
                ViconHipAngle.append(np.float(line[4]))
                
        
    ViconKneeAngle = np.asarray(ViconKneeAngle)
    ViconHipAngle = np.asarray(ViconHipAngle)
    
    data = [ViconKneeAngle,ViconHipAngle]
    
    
if angletype == 'KMA':
    #initialize empty list
    ViconKMAAngle = list([])
    
    #go through csv file and read out the KMA angle
    for line in reader: 
        if len(line) > 0:
    
            if line[0]=='1':
                start=1
            
            if start == 1 and line[0]=='Segments':
                stop=1
                
            if start == 1 and stop == 0:            
                ViconKMAAngle.append(np.float(line[8]))               
                            
    ViconKMAAngle = np.asarray(ViconKMAAngle)    
    
    data = ViconKMAAngle

#save as pkl file
with open(fileout,'wb') as fp:
    pickle.dump(data,fp)