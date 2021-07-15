# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 06:01:04 2020

Loads Load Cell Data, Motion Capture Data and Myosuit data
and brings them to the same timeline

Saves the data in one dict 

@author: Marc
"""

#import packages
import os
from scipy import interpolate as itp
import pickle
import numpy as np
import datetime
import pandas as pd


#define current path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#import functions from other scripts
#folders containing the scripts must be in the same folder then the working folder of this script
from ForceCellProcessing import processData
# from MoCapPostprocessing import telemMS_Vicon
from MoCapPostprocessing import Segmentation
from MoCapPostprocessing import PoseOptimization

#rename function
# ex = telemMS_Vicon.extraction

#set the input files and file locations
# pathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MoCap_Logs\S01_19112020\S01_19112020_Trial02.pkl"
# pathForce = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\ForceCell_Logs\S01_19112020_Force\S01_19112020_T02.csv"
# pathMS    = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MyoSuit_Logs\S01_19112020_MS\S01_19112020_T02.pickle"

#choose if the Myosuit data contains the full IMU information
#True: contains raw IMU data
#False: contains Kalman filtered data
fullIMUdata = False

#determine the name of the output file
# filename_out = 'Testdyn2.pkl'

def data_processing_collect(pathMoCap, pathForce, pathMS, filename_out):
    
    #import Force data
    [Fprox,Fdist,tforce,syncforce]=processData.impcsv(pathForce)
    print('Loading successful: Force sensor data')
    
    #import MoCap marker cloud
    file = open(pathMoCap,'rb')
    MoCapData = pickle.load(file)
    t = MoCapData["time"]
    labelsMoCap = MoCapData["labels"]
    markerpos = MoCapData["points_array"] 
    vgrf = MoCapData["vgrf"]
    
    ##uncomment this for subject 6, changes markers 7 and 8 after frame 9640
    # marker7 = np.vstack([markerpos[:9640,7:8,:],markerpos[9640:,8:9,:]])
    # marker8 = np.vstack([markerpos[:9640,8:9,:],markerpos[9640:,7:8,:]])

    # marker1to6 = markerpos[:,:7,:]
    # marker9to33 = markerpos[:,9:,:]
        
    # markerpos = np.concatenate([marker1to6,marker7,marker8,marker9to33],axis=1)

    
    #calculate total ground reaction force
    Ftot_vert = vgrf[2,:] + vgrf[8,:]
    
    print('Loading successful: Motion Capture Data')
    
    #import MyoSuit data    
    file = open(pathMS,'rb')
    MyoSuitData = pickle.load(file)
    
    syncMS = np.zeros(len(MyoSuitData))
    timeMS = np.zeros(len(MyoSuitData))
    AlphaShank = np.zeros(len(MyoSuitData))
    AlphaThigh = np.zeros(len(MyoSuitData))
    AlphaTrunk = np.zeros(len(MyoSuitData))
    EncCount = np.zeros(len(MyoSuitData))
    CurrentSent = np.zeros(len(MyoSuitData))
    CurrentRead = np.zeros(len(MyoSuitData))
    
    ## K
    R_leg = np.zeros(len(MyoSuitData))
    L_leg = np.zeros(len(MyoSuitData))
    Mode = np.zeros(len(MyoSuitData))
    
    if fullIMUdata == True:
        RS_AccA = np.zeros(len(MyoSuitData))
        RS_AccB = np.zeros(len(MyoSuitData))
        RS_AccC = np.zeros(len(MyoSuitData))
    
        RT_AccA = np.zeros(len(MyoSuitData))
        RT_AccB = np.zeros(len(MyoSuitData))
        RT_AccC = np.zeros(len(MyoSuitData))
    
        Tr_AccA = np.zeros(len(MyoSuitData))
        Tr_AccB = np.zeros(len(MyoSuitData))
        Tr_AccC = np.zeros(len(MyoSuitData))
        
        RS_GyrA = np.zeros(len(MyoSuitData))
        RS_GyrB = np.zeros(len(MyoSuitData))
        RS_GyrC = np.zeros(len(MyoSuitData))
    
        RT_GyrA = np.zeros(len(MyoSuitData))
        RT_GyrB = np.zeros(len(MyoSuitData))
        RT_GyrC = np.zeros(len(MyoSuitData))
    
        Tr_GyrA = np.zeros(len(MyoSuitData))
        Tr_GyrB = np.zeros(len(MyoSuitData))
        Tr_GyrC = np.zeros(len(MyoSuitData))


    #transform the myosuit data to meaningful units
    for i in range(0,len(MyoSuitData)):
        syncMS[i] = MyoSuitData.Dummy06[i]
        timeMS[i]=MyoSuitData.DataTimestamp[i].seconds/1000 #in sec
        AlphaShank[i] = MyoSuitData.RS_AFlt[i]/1000 #in deg
        AlphaThigh[i] = MyoSuitData.RT_AFlt[i]/1000 #in deg
        AlphaTrunk[i] = MyoSuitData.Tr_AFlt[i]/1000 #in deg
        EncCount[i] = MyoSuitData.RM_Enc[i] #integer numbers
        CurrentSent[i] = MyoSuitData.RM_CurS[i]/1000 #in Amp
        CurrentRead[i] = MyoSuitData.RM_CurR[i]/1000  #in Amp
        ##K
        R_leg[i] = MyoSuitData.R_leg[i]
        L_leg[i] = MyoSuitData.L_leg[i]
        Mode[i] = MyoSuitData.Mode[i]
        
        #load additionally the accelerometer and gyroscope data for shank, thigh and trunk segment of the Myosuit
        if fullIMUdata == True:
            RS_AccA[i] = MyoSuitData.RS_AccA[i]
            RS_AccB[i] = MyoSuitData.RS_AccB[i]
            RS_AccC[i] = MyoSuitData.RS_AccC[i]
        
            RT_AccA[i] = MyoSuitData.RT_AccA[i]
            RT_AccB[i] = MyoSuitData.RT_AccB[i]
            RT_AccC[i] = MyoSuitData.RT_AccC[i]
        
            Tr_AccA[i] = MyoSuitData.Tr_AccA[i]
            Tr_AccB[i] = MyoSuitData.Tr_AccB[i]
            Tr_AccC[i] = MyoSuitData.Tr_AccC[i]
            
            RS_GyrA[i] = MyoSuitData.RS_GyrA[i]
            RS_GyrB[i] = MyoSuitData.RS_GyrB[i]
            RS_GyrC[i] = MyoSuitData.RS_GyrC[i]
        
            RT_GyrA[i] = MyoSuitData.RT_GyrA[i]
            RT_GyrB[i] = MyoSuitData.RT_GyrB[i]
            RT_GyrC[i] = MyoSuitData.RT_GyrC[i]
        
            Tr_GyrA[i] = MyoSuitData.Tr_GyrA[i]
            Tr_GyrB[i] = MyoSuitData.Tr_GyrB[i]
            Tr_GyrC[i] = MyoSuitData.Tr_GyrC[i]
            
               
    print('Loading successful: Myosuit data')
    
    #cut off data before sync signal
    #=> cut off force sensor data
    indexforce = syncforce.argmax()
    tforce = tforce[indexforce:]
    tforce -= tforce[0]
    Fdist = Fdist[indexforce:]
    Fprox = Fprox[indexforce:]
    print('\nCut-off successful: Force sensor data')
       
    #=> cut off MyoSuit data
    indexMS = syncMS.argmax()
    
    timeMS = timeMS[indexMS:]-timeMS[indexMS]
    AlphaShank = AlphaShank[indexMS:]
    AlphaThigh = AlphaThigh[indexMS:]
    AlphaTrunk = AlphaTrunk[indexMS:]
    EncCount = EncCount[indexMS:]
    CurrentSent = CurrentSent[indexMS:]
    CurrentRead = CurrentRead[indexMS:]
    
    # K
    R_leg = R_leg[indexMS:]
    L_leg = L_leg[indexMS:]
    Mode = Mode[indexMS:]    

    if fullIMUdata == True:
            RS_AccA = RS_AccA[indexMS:]
            RS_AccB = RS_AccB[indexMS:]
            RS_AccC = RS_AccC[indexMS:]
    
            RT_AccA = RT_AccA[indexMS:]
            RT_AccB = RT_AccB[indexMS:]
            RT_AccC = RT_AccC[indexMS:]
        
            Tr_AccA = Tr_AccA[indexMS:]
            Tr_AccB = Tr_AccB[indexMS:]
            Tr_AccC = Tr_AccC[indexMS:]
            
            RS_GyrA = RS_GyrA[indexMS:]
            RS_GyrB = RS_GyrB[indexMS:]
            RS_GyrC = RS_GyrC[indexMS:]
        
            RT_GyrA = RT_GyrA[indexMS:]
            RT_GyrB = RT_GyrB[indexMS:]
            RT_GyrC = RT_GyrC[indexMS:]
        
            Tr_GyrA = Tr_GyrA[indexMS:]
            Tr_GyrB = Tr_GyrB[indexMS:]
            Tr_GyrC = Tr_GyrC[indexMS:]
                
    print('Cut-off successful: Myosuit data')
    
    
    #interpolate data
    
    #=>interpolate force data
    interFprox = itp.interp1d(tforce,Fprox)
    interFdist = itp.interp1d(tforce,Fdist)
    Fprox = interFprox(t)
    Fdist = interFdist(t)
    
    print('\nInterpolation successful: Force sensor data')
    
    #=>interpolate MyoSuit Data
    interAShank = itp.interp1d(timeMS,AlphaShank)
    interAThigh = itp.interp1d(timeMS,AlphaThigh)
    interATrunk = itp.interp1d(timeMS,AlphaTrunk)
    interEC     = itp.interp1d(timeMS,EncCount)
    interCR     = itp.interp1d(timeMS,CurrentRead)
    interCS     = itp.interp1d(timeMS,CurrentSent)
    ##K 
    interR_leg = itp.interp1d(timeMS, R_leg)
    interL_leg = itp.interp1d(timeMS, L_leg)
    interMode = itp.interp1d(timeMS, Mode)

    
    AlphaShank = interAShank(t)
    AlphaThigh = interAThigh(t)
    AlphaTrunk = interATrunk(t)
    EncCount   = interEC(t)
    CurrentRead = interCR(t)
    CurrentSent = interCS(t)  
    ##K
    R_Leg = interR_leg(t)
    L_Leg = interL_leg(t)
    Mode = interMode(t)
    
    pdframe = pd.DataFrame()
    if fullIMUdata == True:
            IntRS_AccA = itp.interp1d(timeMS,RS_AccA)
            IntRS_AccB = itp.interp1d(timeMS,RS_AccB)
            IntRS_AccC = itp.interp1d(timeMS,RS_AccC)
    
            IntRT_AccA = itp.interp1d(timeMS,RT_AccA)
            IntRT_AccB = itp.interp1d(timeMS,RT_AccB)
            IntRT_AccC = itp.interp1d(timeMS,RT_AccC)
        
            IntTr_AccA = itp.interp1d(timeMS,Tr_AccA)
            IntTr_AccB = itp.interp1d(timeMS,Tr_AccB)
            IntTr_AccC = itp.interp1d(timeMS,Tr_AccC)
            
            IntRS_GyrA = itp.interp1d(timeMS,RS_GyrA)
            IntRS_GyrB = itp.interp1d(timeMS,RS_GyrB)
            IntRS_GyrC = itp.interp1d(timeMS,RS_GyrC)
        
            IntRT_GyrA = itp.interp1d(timeMS,RT_GyrA)
            IntRT_GyrB = itp.interp1d(timeMS,RT_GyrB)
            IntRT_GyrC = itp.interp1d(timeMS,RT_GyrC)
        
            IntTr_GyrA = itp.interp1d(timeMS,Tr_GyrA)
            IntTr_GyrB = itp.interp1d(timeMS,Tr_GyrB)
            IntTr_GyrC = itp.interp1d(timeMS,Tr_GyrC)
            
            RS_AccA = IntRS_AccA(t)
            RS_AccB = IntRS_AccB(t)
            RS_AccC = IntRS_AccC(t)
    
            RT_AccA = IntRT_AccA(t)
            RT_AccB = IntRT_AccB(t)
            RT_AccC = IntRT_AccC(t)
        
            Tr_AccA = IntTr_AccA(t)
            Tr_AccB = IntTr_AccB(t)
            Tr_AccC = IntTr_AccC(t)
            
            RS_GyrA = IntRS_GyrA(t)
            RS_GyrB = IntRS_GyrB(t)
            RS_GyrC = IntRS_GyrC(t)
        
            RT_GyrA = IntRT_GyrA(t)
            RT_GyrB = IntRT_GyrB(t)
            RT_GyrC = IntRT_GyrC(t)
        
            Tr_GyrA = IntTr_GyrA(t)
            Tr_GyrB = IntTr_GyrB(t)
            Tr_GyrC = IntTr_GyrC(t)
            
            pddata = {'RS_AccA':RS_AccA,'RS_AccB':RS_AccB,'RS_AccC':RS_AccC,
                      'RT_AccA':RT_AccA,'RT_AccB':RT_AccB,'RT_AccC':RT_AccC,
                      'Tr_AccA':Tr_AccA,'Tr_AccB':Tr_AccB,'Tr_AccC':Tr_AccC,
                      'RS_GyrA':RS_GyrA,'RS_GyrB':RS_GyrB,'RS_GyrC':RS_GyrC,
                      'RT_GyrA':RT_GyrA,'RT_GyrB':RT_GyrB,'RT_GyrC':RT_GyrC,
                      'Tr_GyrA':Tr_GyrA,'Tr_GyrB':Tr_GyrB,'Tr_GyrC':Tr_GyrC}
            
            pdframe = pd.DataFrame(data=pddata)
            
    #downsample vertical ground reaction force
    Ftot_vert_corr = np.zeros(len(t))
    for i in range(0,len(t)):
        if 10*i+10 > len(Ftot_vert):
            Ftot_vert_corr[i]=-np.sum(Ftot_vert[10*i:])/(len(Ftot_vert)-10*i)
        elif 10*i+10 <= len(Ftot_vert):
            Ftot_vert_corr[i]=-np.sum(Ftot_vert[10*i:10*i+10])/10
        
    
    print('Interpolation successful: Myosuit data')

    
    #split MoCap data into segments
    segments = Segmentation.segmentation(markerpos,labelsMoCap)
    print("\nSegmentation successful")
    
    #find reference postures for each segment
    references = []
    
    for segment in segments:
        no = 0.
        avg = np.zeros_like(segment[0])
        
        #average the entries for each segment for the first 500 frames
        for frame in segment[0:500]:
            if np.any(frame[:,0]==0)==False:
                avg += frame
                no += 1.
        ref = avg / no
        
        references.append(ref)
        
    del no
    del ref
    del avg
    del frame
    del segment
        
    
    #correct MoCap markers
    
    #time estimation and ask for proceeding    
    optnumber = len(segments)*len(segments[0])
    esttimes = optnumber * 0.0002
    esttimeh = esttimes / 60
    
    print('\nEstimated time:',round(esttimeh,1),'minutes')
    proceed = input("Do you want to proceed? (y/n)")
    
    #set up empty lists
    segmentscorr = []
    res_norm_allseg = []
    princ_dir_allseg = []
        
    if proceed =="y":
        # print('\nOptimization procedure...')
        
        #set up counter indicating the processed segment
        segno = 0
        
        #start timer
        starttime = datetime.datetime.now()
        
        #iteration over all segments
        for segment in segments:
            
            #provide optimization progress to console
            segno +=1
            # print('\nOptimizing Segment',segno)
            
            #correct the segment data
            segcorr,res_norm,princ_dir = PoseOptimization.optimsegment_Soederkvist(references[segno-1],segment)
            segmentscorr.append(segcorr)
            res_norm_allseg.append(res_norm)
            princ_dir_allseg.append(princ_dir)
         
        #stop timer
        duration = datetime.datetime.now()-starttime
    
    if proceed =="n":
        print('\nno optimization done')
        
    ##K
    
    lenMS = len(t)
    lenMC = len(segmentscorr[0])
    lenF = len(Ftot_vert_corr)
    len_corr = min(lenMS, lenMC, lenF)
    
    out_dict = {}
    
    out_dict["time"] = t
    out_dict["AlphaShank"] = AlphaShank[0: len_corr]
    out_dict["AlphaThigh"] = AlphaThigh[0: len_corr]
    out_dict["AlphaTrunk"] = AlphaTrunk[0: len_corr]
    out_dict["EncCount"] = EncCount[0: len_corr]
    out_dict["CurrentRead"] = CurrentRead[0: len_corr]
    out_dict["CurrentSent"] = CurrentSent[0: len_corr]
    out_dict["Fprox"] = Fprox[0: len_corr]
    out_dict["Fdist"] = Fdist[0: len_corr]
    out_dict["R_leg"] = R_Leg[0: len_corr]
    out_dict["L_leg"] = L_Leg[0: len_corr]
    out_dict["Mode"] = Mode[0: len_corr]
    out_dict["vgrf"] = Ftot_vert_corr[0: len_corr]
    
    # ignore labelsMoCap et markerpos - pas utilisé (vu que info est transféré 
    # ds segment) sauf pr la longueur pr la suite - trouver autre façon
    
    # ignore principal directions aussi 
    
    out_dict["ShankSegment"] = segmentscorr[0][0: len_corr]
    out_dict["ThighSegment"] = segmentscorr[1][0: len_corr]
    out_dict["HipSegment"] = segmentscorr[2][0: len_corr]
    out_dict["KMAlowSegment"] = segmentscorr[3][0: len_corr]
    out_dict["KMAupSegment"] = segmentscorr[4][0: len_corr]
    out_dict["TDUSegment"] = segmentscorr[5][0: len_corr]
    
    out_dict["ResNormShank"] = res_norm_allseg[0][0: len_corr]
    out_dict["ResNormThigh"] = res_norm_allseg[1][0: len_corr]
    out_dict["ResNormHip"] = res_norm_allseg[2][0: len_corr]
    out_dict["ResNormKMAlow"] = res_norm_allseg[3][0: len_corr]
    out_dict["ResNormKMAup"] = res_norm_allseg[4][0: len_corr]
    out_dict["ResNormTDU"] = res_norm_allseg[5][0: len_corr]

    # if save_file == True:
    #     os.chdir(dname)
    #     def save_obj(obj, name ):
    #         with open(name, 'wb') as f:
    #             pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
                
    #     save_obj(out_dict, filename_out)
    #     print('\n...data saved to file')
   
    print("Data Processing Collect end")
    
    return out_dict
    
    
    


    
    
    

    
    
    
    
    
    
    
    