# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 07:01:31 2020

@author: Marc
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

#path of vicon leg angles
Vicloc = r"C:\Users\wirth\Documents\MScBBrep\MoCapPostprocessing\S06_23112020_T01_ViconAngles.pkl"

#path of mocap leg angles
MoCloc = r"C:\Users\wirth\polybox\MSc\Master Thesis\03_Relevant Files\03_Software\PostprocessedFiles\S06_23112020_T01\S06_23112020_T01_Interpreted_corr2.pkl"

#angletype: 'Leg'  / 'KMA'
angletype = 'Leg'

#open files
with open(Vicloc,'rb') as fileopen:
    VicAngles=pickle.load(fileopen)
with open(MoCloc,'rb') as fileopen:
    MCdata=pickle.load(fileopen)

if angletype == 'Leg':
    VicKneeAng = VicAngles[0]
    VicHipAng  = VicAngles[1]
    
    CustKneeAng = MCdata[6][0:len(VicKneeAng)]
    CustHipAng  = MCdata[7][0:len(VicKneeAng)]
    
    #correct data offset
    CKAcorr = CustKneeAng - CustKneeAng[0] + VicKneeAng[0]
    CHAcorr = CustHipAng - CustHipAng[0] + VicHipAng[0]
    
    #get the measurement difference
    deltaKnee = CKAcorr - VicKneeAng
    
    #draw plot to compare the VICON angles with the custom processed angles
    fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,1]})

    
    ax[0].plot(CKAcorr[20700:21860],label='Custom Knee Angle')
    ax[0].plot(VicKneeAng[20700:21860],label='VICON Knee Angle')
    ax[0].set_ylabel('Knee Angle [°]',fontsize=24,weight='bold')
    ax[0].tick_params(axis='x',labelsize=24)
    ax[0].tick_params(axis='y',labelsize=24)
    ax[0].legend(fontsize=24    )
    ax[0].grid()    
    ax[0].set_title('Postprocessing Comparison',fontsize=28,weight='bold')
    
    ax[1].plot(deltaKnee[20700:21860])
    ax[1].set_xlabel('Frame number',fontsize=24,weight='bold')
    ax[1].set_ylabel('Angle Difference [°]',fontsize=20,weight='bold')
    ax[1].tick_params(axis='x',labelsize=24)
    ax[1].tick_params(axis='y',labelsize=24)
    ax[1].grid()

if angletype == 'KMA':
    ViKMA = VicAngles
        
    CustKMAAng = MCdata[11][0:len(ViKMA)]
    
    #correct data offset  
    CustKMAcorr = CustKMAAng - CustKMAAng[0] + ViKMA[0]
    
    #get the measurement difference
    deltaKMA = CustKMAcorr - ViKMA
    
    #draw plot to compare the VICON angles with the custom processed angles
    fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,1]})
    
    ax[0].plot(CustKMAcorr,label='Custom KMA Angle')
    ax[0].plot(ViKMA,label='VICON KMA Angle')
    ax[0].set_xlabel('Frame number')
    ax[0].set_ylabel('KMA Angle [°]')
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(deltaKMA)
    ax[1].set_xlabel('Frame number')
    ax[1].set_ylabel('Angle Diference [°]')
    ax[1].grid()

