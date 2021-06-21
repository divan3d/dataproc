# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:35:38 2021

@author: ksdiv

data pre- processing - interprets mocap angles
saves all needed data in pd dataframe 
call function "data_processing_interpretation(name_filein, name_fileout, save_file)"
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

# #define current path
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

# ### prit de InterpretationPipeline - coupé et modifié 

# def op_pickle(file):
#     with open(file,'rb') as fileopen:
#         data=pickle.load(fileopen)
#         return data

# def save_obj(obj, name ):
#     with open(name, 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
   

#function to calculate the initial direction of the coordinate frames assigned to the segments
def getinitialdirections(segments):
    
    #define the direction of the reference vectors in the first frame
    shankinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    thighinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    trunkinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    KMAlowinit = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    KMAupinit  = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
    TDUinit    = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=float)
        
        
    #change z-vector of shank segment to be aligned along the connection of the two markers lying the most apart from each other
    shankinit[:,2] = (segments[0][0][2,:]-segments[0][0][0,:])/np.linalg.norm(segments[0][0][2,:]-segments[0][0][0,:])
    
    #combine the segment initial directions to one list
    initvecs = np.array([shankinit,thighinit,trunkinit,KMAlowinit,KMAupinit,TDUinit])
    
    #toggle, if the initial directions for the Myosuit shall be aligned to the directions of the IMU's
    #recommmended for most cases, checking data afterwards is necessary to ensure correct data processing
    alignMyosuitInitVecs = True
    
    if alignMyosuitInitVecs == True:
        #manual definition of the markers placed on the Myosuit, read out from CAD data
        KMAlcoord  = np.array([[-6.7,8.99225,-33.3665],[103.247,-27.3496,-29.9716],[-6.9,-8.48,96.752]])
        KMAucoord  = np.array([[113.951,-81.7746,51.9328],[-7.15,13.12,7.5],[-10.6182,-7.02642,101.942]])
        TDUcoord   = np.array([[-145.042,14.85,33.7003],[0,64.9066,-48.3448],[145.042,14.85,33.7003]])
        
        #calculate reference frames for each Myosuit segment
        references = []    
        for i in range(3,6):
            no = 0.
            avg = np.zeros_like(segments[i][0])
            for frame in segments[i][0:500]:
                for j in range(0,len(segments[i][0])):
                    if np.count_nonzero(frame[j,:])!=0:
                        avg += frame
                        no += 1.
            ref = avg / no
            references.append(ref)
        
        #transform the read out coordinates to the reference frame
        KMAltransformed = PoseOptimization.optimize(KMAlcoord,references[0])
        KMAutransformed = PoseOptimization.optimize(KMAucoord,references[1])
        TDUtransformed  = PoseOptimization.optimize(TDUcoord ,references[2])
        
        #combine the initial coordinates and the transformed coordinates to a temporary segment
        KMAlsegtemp = [KMAlcoord,KMAltransformed]
        KMAusegtemp = [KMAucoord,KMAutransformed]
        TDUsegtemp  = [TDUcoord ,TDUtransformed]
        
        #essayer ça -- en tout cas code tourne 
        # KMAlsegtemp = [KMAltransformed]
        # KMAusegtemp = [KMAutransformed]
        # TDUsegtemp  = [TDUtransformed]
        
        #build up Myosuit structure
        segtemp = [KMAlsegtemp,KMAusegtemp,TDUsegtemp]
        
        #build a frame of initial vectors for the Myosuit segments
        inittemp = np.array([KMAlowinit,KMAupinit,TDUinit])
        
        #get directions of the coordinate frame in the transformed marker cloud when the coordinate system is given for the untransformed markers
        transformedvec = getdirections(segtemp,inittemp)
        
        #define the the transformed directions to be the initial directions
        initvecs = np.array([shankinit,thighinit,trunkinit,transformedvec[0][1],transformedvec[1][1],transformedvec[2][1]])
        # initvecs = np.array([shankinit,thighinit,trunkinit,transformedvec[0][0],transformedvec[1][0],transformedvec[2][0]])
    
    
    return initvecs
    
#calculate the direction of the unit vectors of the coordinate systems in each segment based on the initial configuration
def getdirections(segments,initvecs):
        
    #initialization of unit vector set for each frame in each segment        
    uvecs = []
     
    #determine unit vectors for each frame        
    for segment in segments:
        uvecseg = []
        for frame in segment:
            uvecset = np.zeros([3,3])
            #vector 1; going from marker 0 to marker 1
            uvecset[:,0] = (frame[1,:]-frame[0,:])/np.linalg.norm(frame[1,:]-frame[0,:])
            #vector 2; going from marker  0 to marker 2
            uvecset[:,1] = (frame[2,:]-frame[0,:])/np.linalg.norm(frame[2,:]-frame[0,:])
            #vector 3; orthogonal to vector 1 and 2
            uvecset[:,2] = np.cross(uvecset[:,0],uvecset[:,1])/np.linalg.norm(np.cross(uvecset[:,0],uvecset[:,1]))
                
            uvecseg.append(uvecset)
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
        KMAlowang[i] = m.asin(refvecs[3][i][1,2])*180/m.pi + 90
        KMAupang[i]  = m.asin(refvecs[4][i][1,2])*180/m.pi + 90
               
            
        MoCapAngles = [shankang,thighang,KMAlowang,KMAupang]
        
    return MoCapAngles



def data_processing_interpretation(data_test):
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
    check_mode = statistics.mean(data_test["Mode"])
    print("MyoSuit mode check : %f" % check_mode)
        
    if not "ShankSegment" in data_test.keys():
        print("No segment data, no angle calculation posssible")
        sys.exit()
        
    #define the initial direction of the coordinate systems of each segment
    # moins chiant de faire ça que de ré-écrire les fonctions qui utilisent
    segments = []
    segments.append(data_test["ShankSegment"])
    segments.append(data_test["ThighSegment"])
    segments.append(data_test["HipSegment"])
    segments.append(data_test["KMAlowSegment"])
    segments.append(data_test["KMAupSegment"])    
    segments.append(data_test["TDUSegment"])
    
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
    temp["force"] = data_test["Fprox"]
    temp["res_norm_shank"] = data_test["ResNormShank"]
    temp["res_norm_thigh"] = data_test["ResNormThigh"]
    temp["res_norm_kmal"] = data_test["ResNormKMAlow"]
    temp["res_norm_kmau"] = data_test["ResNormKMAup"]
    temp["current_sent"] = data_test["CurrentSent"]
    temp["current_read"] = data_test["CurrentRead"]
    temp["L_leg"] = data_test["L_leg"]
    temp["R_leg"] = data_test["R_leg"]
    
    df = pd.DataFrame(data = temp)
    
    # df.to_pickle(fileout)
    
    # if save_file == True:
    #     os.chdir(dname)
    #     save_obj(df, fileout)
    #     print("data saved to file")
    
    print("data processing interpretation end")
    
    return df
#%%

# filein = r"E:\ETHZ\mast_sem_IV\pdm\marc\code\marcwirththesis\Testdyn2.pkl"
# fileout = "TEST2_Postprocessed_Mocap_angles.pkl"

# t_df = data_processing_interpretation(filein, fileout)  