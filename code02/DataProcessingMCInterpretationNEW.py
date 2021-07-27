# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:35:38 2021

@author: ksdiv

data pre- processing - interprets mocap angles
saves all needed data in pd dataframe 
call function "data_processing_interpretation(name_filein, have MS)"
"""

import os 
import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
from MoCapPostprocessing import PoseOptimization
from scipy import interpolate as itp
import pickle
import sys
import pandas as pd
import statistics


   

#function to calculate the initial direction of the coordinate frames assigned to the segments
def getinitialdirections(segments):
    
    #define the direction of the reference vectors in the first frame
    shankinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    thighinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    KMAlowinit = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    KMAupinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
        
        
    # shank z vect : marker 4 to 1 
    # ressemble le plus a comment marc a fait
    shankinit[:,2] = (segments[0][0][0,:]-segments[0][0][3,:])/np.linalg.norm(segments[0][0][0,:]-segments[0][0][3,:])
    # thigh z vect : marker 4 to 1 
    thighinit[:,2] = (segments[1][0][0,:]-segments[1][0][3,:])/np.linalg.norm(segments[1][0][0,:]-segments[1][0][3,:])
    # kmal z vect : marker 3 to 1 
    KMAlowinit[:,2] = (segments[2][0][0,:]-segments[2][0][2,:])/np.linalg.norm(segments[2][0][0,:]-segments[2][0][2,:])
    # kmau z vect : marker 4 to 1 
    KMAupinit[:,2] = (segments[3][0][0,:]-segments[3][0][1,:])/np.linalg.norm(segments[3][0][0,:]-segments[3][0][1,:])
    
    #combine the segment initial directions to one list
    initvecs = np.array([shankinit,thighinit,KMAlowinit,KMAupinit])
    
        
    return initvecs
    
#calculate the direction of the unit vectors of the coordinate systems in each segment based on the initial configuration
def getdirections(segments,initvecs):
        
    #initialization of unit vector set for each frame in each segment        
    uvecs = []
     
    #determine unit vectors for each frame 
    counter = 0        
    for segment in segments:
        uvecseg = []
        for frame in segment:
            if counter == 0: # shank
                uvecset = np.zeros([3,3])
                #vector 1; going from marker 5 to marker 4
                uvecset[:,0] = (frame[3,:]-frame[4,:])/np.linalg.norm(frame[3,:]-frame[4,:])
                #vector 2; going from marker  5 to marker 3
                uvecset[:,1] = (frame[2,:]-frame[4,:])/np.linalg.norm(frame[2,:]-frame[4,:])
                #vector 3; orthogonal to vector 1 and 2
                uvecset[:,2] = np.cross(uvecset[:,0],uvecset[:,1])/np.linalg.norm(np.cross(uvecset[:,0],uvecset[:,1]))
                    
                uvecseg.append(uvecset)
            else :
                uvecset = np.zeros([3,3])
                #vector 1; going from marker 0 to marker 1
                uvecset[:,0] = (frame[1,:]-frame[0,:])/np.linalg.norm(frame[1,:]-frame[0,:])
                #vector 2; going from marker  0 to marker 2
                uvecset[:,1] = (frame[2,:]-frame[0,:])/np.linalg.norm(frame[2,:]-frame[0,:])
                #vector 3; orthogonal to vector 1 and 2
                uvecset[:,2] = np.cross(uvecset[:,0],uvecset[:,1])/np.linalg.norm(np.cross(uvecset[:,0],uvecset[:,1]))
                    
                uvecseg.append(uvecset)
        
        
        counter = counter + 1
        uvecs.append(uvecseg)     
     
    #determine coefficients to relate unit vectors and reference
    coeffs = []
        
    for i in range(0,len(uvecs)):
        uvecavg = np.nanmean(np.asarray(uvecs[i][0:100]),axis=0)
        C = np.zeros([3,3])
        C = np.matmul(np.linalg.inv(uvecavg),initvecs[i])
        
        coeffs.append(C)
    
    #determine reference vectors for each consecutive frame
    refvecs = []
    count = 0
    
    for uvecseg in uvecs:
        refvecseg = []
        for vec in uvecseg:
            temp = np.matmul(vec,coeffs[count])
            for i in range(0,3):
                temp[:,i]=temp[:,i]/np.linalg.norm(temp[:,i])
            refvecseg.append(temp)
        refvecs.append(refvecseg)
        count += 1
        
    return refvecs

#calculate the segment and relative angles based on the mocap measurements
def getMoCapAngles(initvecs,refvecs,segments):
    
    #initialize empty angle vectors
    shankang     = np.zeros([len(refvecs[0])])
    thighang     = np.zeros([len(refvecs[0])])
    KMAlowang    = np.zeros([len(refvecs[0])])
    KMAupang     = np.zeros([len(refvecs[0])])
  

    for i in range(0,len(refvecs[0])):
                                 

        #determine segment angles relative to the z-vector in the initial frame
        #90° are added to have the angles related to the ground
        shankang[i]  = m.asin(refvecs[0][i][1,2])*180/m.pi + 90
        thighang[i]  = m.asin(refvecs[1][i][1,2])*180/m.pi + 90
        KMAlowang[i] = m.asin(refvecs[2][i][1,2])*180/m.pi + 90
        KMAupang[i]  = m.asin(refvecs[3][i][1,2])*180/m.pi + 90
               
            
        MoCapAngles = [shankang,thighang,KMAlowang,KMAupang]
        
    return MoCapAngles



def data_processing_interpretation(data_test, have_MS_data):
    """
    interprets MoCap angles for thigh, shank, KMAU, KMAL and saves all other
    data that could be useful into pd dtaframe

    Parameters
    ----------
    filein : str : path of file that comes out of data_processing_collect 
    fileout : str : name of output dataframe
    save_file : bool : wether or not to save the dictionary 

    Returns
    -------
    df : pd dataframe : contains info for dyn or static experiment 

    """
    
    print("data processing interpretation start")
        
           
    # check if mode of myosuit is on 1 - if mean value = 1 means that myosuit mode 
    # was on concentric the whole time
    if have_MS_data == True : 
        check_mode = statistics.mean(data_test["Mode"])
        print("MyoSuit mode check : %f" % check_mode)
        
    if not "ShankSegment" in data_test.keys():
        print("No segment data, no angle calculation posssible")
        sys.exit()
        
    #define the initial direction of the coordinate systems of each segment
    # moins chiant de faire ça que de ré-écrire les fonctions qui utilisent
    
    # si change l'ordre faut aussi changer dans getinitialdirections
    
    segments = []
    segments.append(data_test["ShankSegment"])
    segments.append(data_test["ThighSegment"])
    segments.append(data_test["KMAlowSegment"])
    segments.append(data_test["KMAupSegment"])    
    # segments.append(data_test["TDUSegment"])
    # segments.append(data_test["ShouldersSegment"])
     
    
    initvecs = getinitialdirections(segments)
    print('Initial segment directions calculated')
    
    #get the direction of the coordinate systems in each frame
    refvecs = getdirections(segments,initvecs)
    print('Segment directions for all frames calculated')
    
    #get the segment angles recorded in Motion Capture
    MoCapAngles = getMoCapAngles(initvecs,refvecs,segments)
    print('Segment Angles based on Motion Capture data is calculated')
    
    
    # save to pd data frame 
    
    temp = {}
    
    temp["t"] = data_test["time"]
    temp["mc_shank_angle"] = MoCapAngles[0]
    temp["mc_thigh_angle"] = MoCapAngles[1]
    temp["mc_kmal_angle"] = MoCapAngles[2]
    temp["mc_kmau_angle"] = MoCapAngles[3]
    temp["vgrf"] = data_test["vgrf"]
    temp["res_norm_shank"] = data_test["ResNormShank"]
    temp["res_norm_thigh"] = data_test["ResNormThigh"]
    temp["res_norm_kmal"] = data_test["ResNormKMAlow"]
    temp["res_norm_kmau"] = data_test["ResNormKMAup"]
    
    if have_MS_data == True : 
        temp["current_sent"] = data_test["CurrentSent"]
        temp["current_read"] = data_test["CurrentRead"]
        temp["L_leg"] = data_test["L_leg"]
        temp["R_leg"] = data_test["R_leg"]
        temp["AlphaShank"] = data_test["AlphaShank"]
        temp["AlphaThigh"] = data_test["AlphaThigh"]
        temp["AlphaTrunk"] = data_test["AlphaTrunk"]
        temp["Mode"] = data_test["Mode"]
        temp["Force"] = data_test["Force"]
        temp["ForceLevel"] = data_test["ForceLevel"]
        temp["GyroCThigh"] = data_test["GyroCThigh"]
        temp["GyroCShank"] = data_test["GyroCShank"]
    
    df = pd.DataFrame(data = temp)
    
    
    print("data processing interpretation end")
    
    return df

