# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:21 2021

@author: ksdiv
"""


#import packages
import os
from scipy import interpolate as itp
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from MoCapPostprocessing import SegmentationNew
from MoCapPostprocessing import PoseOptimization

# nouveau !! 
os.chdir(dname)

def data_processing_collect(pathMoCap, pathMS, have_MS_data):
    
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
    vgrf1 = vgrf[2,:]
    vgrf2 = vgrf[8,:]
    
    print('Loading successful: Motion Capture Data')
    
    if have_MS_data == True : 
    
        #import MyoSuit data    
        file = open(pathMS,'rb')
        MyoSuitData = pickle.load(file)
        
        syncMS = np.zeros(len(MyoSuitData))
        timeMS = np.zeros(len(MyoSuitData))
        AlphaShank = np.zeros(len(MyoSuitData))
        AlphaThigh = np.zeros(len(MyoSuitData))
        AlphaTrunk = np.zeros(len(MyoSuitData))
        # EncCount = np.zeros(len(MyoSuitData))
        CurrentSent = np.zeros(len(MyoSuitData))
        CurrentRead = np.zeros(len(MyoSuitData))
        R_leg = np.zeros(len(MyoSuitData))
        L_leg = np.zeros(len(MyoSuitData))
        Mode = np.zeros(len(MyoSuitData))
        Force = np.zeros(len(MyoSuitData))
        ForceLevel = np.zeros(len(MyoSuitData))
        GyroCThigh = np.zeros(len(MyoSuitData))
        GyroCShank = np.zeros(len(MyoSuitData))
        AccelAThigh = np.zeros(len(MyoSuitData))
        AccelAShank = np.zeros(len(MyoSuitData))
        
        #Andri 
        GyroAThigh = np.zeros(len(MyoSuitData))
        GyroAShank = np.zeros(len(MyoSuitData))
        GyroBThigh = np.zeros(len(MyoSuitData))
        GyroBShank = np.zeros(len(MyoSuitData))
        AccelBThigh = np.zeros(len(MyoSuitData))
        AccelBShank = np.zeros(len(MyoSuitData))
        AccelCThigh = np.zeros(len(MyoSuitData))
        AccelCShank = np.zeros(len(MyoSuitData))
        
        #Hall sensors
        HallSensor = np.zeros(len(MyoSuitData))
        
        # Time 
        Time = MyoSuitData.tv_sec + MyoSuitData.tv_nsec * 1e-9
        
        # Mode_temp
        sysinf = MyoSuitData.sensorS_StateSysInf - 8
        Mode_temp = sysinf * MyoSuitData.sensorS_StateSysMode
        Mode_temp = Mode_temp.replace(0,3)
    
    
        #transform the myosuit data to meaningful units
        for i in range(0,len(MyoSuitData)):
            syncMS[i] = MyoSuitData.extSensorS_SyncIn[i]
            timeMS[i]= Time[i] #in sec
            AlphaShank[i] = MyoSuitData.IMURightShankKalmanAngle[i]/1000 #in deg
            AlphaThigh[i] = MyoSuitData.IMURightThighKalmanAngle[i]/1000 #in deg
            AlphaTrunk[i] = MyoSuitData.IMUTrunkKalmanAngle[i]/1000 #in deg
            # don't have encoder
            # EncCount[i] = MyoSuitData.RM_Enc[i] #integer numbers - faudra utiliser hall sensors ? 
            CurrentSent[i] = MyoSuitData.CurrentRightMotor[i]/1000 #in Amp
            CurrentRead[i] = MyoSuitData.ActualCurrentRightMotor[i]/1000  #in Amp
            ##K
            R_leg[i] = MyoSuitData.sensorS_StateRightStanceDetection[i]
            L_leg[i] = MyoSuitData.sensorS_StateLeftStanceDetection[i]
            Mode[i] = Mode_temp[i]
            Force[i] = MyoSuitData.extSensorS_Analog0[i]/1000 # pk pas, valeurs enormes sinon
            ForceLevel[i] = MyoSuitData.sensorS_StateRightForce[i] 
            GyroCThigh[i] = MyoSuitData.IMURightThighGyroC[i]*0.0175
            GyroCShank[i] = MyoSuitData.IMURightShankGyroC[i]*0.0175
            AccelAThigh[i] = MyoSuitData.IMURightThighAccelA[i]/4096
            AccelAShank[i] = MyoSuitData.IMURightShankAccelA[i]/4096
            
            # Andri 
            GyroAThigh[i] = MyoSuitData.IMURightThighGyroA[i]*0.0175
            GyroAShank[i] = MyoSuitData.IMURightShankGyroA[i]*0.0175
            AccelBThigh[i] = MyoSuitData.IMURightThighAccelB[i]/4096 
            AccelBShank[i] = MyoSuitData.IMURightShankAccelB[i]/4096
            GyroBThigh[i] = MyoSuitData.IMURightThighGyroB[i]*0.0175
            GyroBShank[i] = MyoSuitData.IMURightShankGyroB[i]*0.0175
            AccelCThigh[i] = MyoSuitData.IMURightThighAccelC[i]/4096 
            AccelCShank[i] = MyoSuitData.IMURightShankAccelC[i]/4096
            
            # Hall Sensors
            HallSensor[i] = MyoSuitData.HallRightMotor[i]
                
                   
        print('Loading successful: Myosuit data')
     
        # SE_FL5_1
        # syncMS[6194] = 1       
        
        # SE_FL3
        # syncMS[3232] = 1       
     
        #=> cut off MyoSuit data
        indexMS = syncMS.argmax()
        
        timeMS = timeMS[indexMS:]-timeMS[indexMS]
        AlphaShank = AlphaShank[indexMS:]
        AlphaThigh = AlphaThigh[indexMS:]
        AlphaTrunk = AlphaTrunk[indexMS:]
        # EncCount = EncCount[indexMS:]
        CurrentSent = CurrentSent[indexMS:]
        CurrentRead = CurrentRead[indexMS:]
        R_leg = R_leg[indexMS:]
        L_leg = L_leg[indexMS:]
        Mode = Mode[indexMS:]    
        Force = Force[indexMS:]
        ForceLevel = ForceLevel[indexMS:]
        GyroCThigh = GyroCThigh[indexMS:]
        GyroCShank = GyroCShank[indexMS:]
        AccelAThigh = AccelAThigh[indexMS:]
        AccelAShank = AccelAShank[indexMS:]
        
        # Andri 
        GyroAThigh = GyroAThigh[indexMS:]
        GyroAShank = GyroAShank[indexMS:]
        AccelBThigh = AccelBThigh[indexMS:]
        AccelBShank = AccelBShank[indexMS:]
        GyroBThigh = GyroBThigh[indexMS:]
        GyroBShank = GyroBShank[indexMS:]
        AccelCThigh = AccelCThigh[indexMS:]
        AccelCShank = AccelCShank[indexMS:]
        
        HallSensor = HallSensor[indexMS:]
                    
        print('Cut-off successful: Myosuit data')
        
        
        #interpolate data
        
        
        #=>interpolate MyoSuit Data
        interAShank = itp.interp1d(timeMS,AlphaShank)
        interAThigh = itp.interp1d(timeMS,AlphaThigh)
        interATrunk = itp.interp1d(timeMS,AlphaTrunk)
        # interEC     = itp.interp1d(timeMS,EncCount)
        interCR     = itp.interp1d(timeMS,CurrentRead)
        interCS     = itp.interp1d(timeMS,CurrentSent)
        ##K 
        interR_leg = itp.interp1d(timeMS, R_leg)
        interL_leg = itp.interp1d(timeMS, L_leg)
        interMode = itp.interp1d(timeMS, Mode)
        interForce = itp.interp1d(timeMS, Force)
        interFL = itp.interp1d(timeMS, ForceLevel)
        interGCT = itp.interp1d(timeMS,GyroCThigh)
        interGCS = itp.interp1d(timeMS,GyroCShank)
        interAAT = itp.interp1d(timeMS,AccelAThigh)
        interAAS = itp.interp1d(timeMS,AccelAShank)
        
        #Andri 
        interGAT = itp.interp1d(timeMS,GyroAThigh)
        interGAS = itp.interp1d(timeMS,GyroAShank)
        interABT = itp.interp1d(timeMS,AccelBThigh)
        interABS = itp.interp1d(timeMS,AccelBShank)
        interGBT = itp.interp1d(timeMS,GyroBThigh)
        interGBS = itp.interp1d(timeMS,GyroBShank)
        interACT = itp.interp1d(timeMS,AccelCThigh)
        interACS = itp.interp1d(timeMS,AccelCShank)
        
        interHS = itp.interp1d(timeMS, HallSensor)
    
        # c'était necessaire pr SD_FL5, parce que données MS se coupent soudainement
        t = t[t<timeMS[-1]]
            
    
        AlphaShank = interAShank(t)
        AlphaThigh = interAThigh(t)
        AlphaTrunk = interATrunk(t)
        # EncCount   = interEC(t)
        CurrentRead = interCR(t)
        CurrentSent = interCS(t)  
        R_Leg = interR_leg(t)
        L_Leg = interL_leg(t)
        Mode = interMode(t)
        Force = interForce(t)
        ForceLevel = interFL(t)
        GyroCShank = interGCS(t)
        GyroCThigh = interGCT(t)
        AccelAShank = interAAS(t)
        AccelAThigh = interAAT(t)
        
        # andri 
        GyroAShank = interGAS(t)
        GyroAThigh = interGAT(t)
        AccelBShank = interABS(t)
        AccelBThigh = interABT(t)
        GyroBShank = interGBS(t)
        GyroBThigh = interGBT(t)
        AccelCShank = interACS(t)
        AccelCThigh = interACT(t)
        
        HallSensor = interHS(t)
        
        print('Interpolation successful: Myosuit data')
    
            
    #downsample vertical ground reaction force
    Ftot_vert_corr = np.zeros(len(t))
    for i in range(0,len(t)):
        if 10*i+10 > len(Ftot_vert):
            Ftot_vert_corr[i]=-np.sum(Ftot_vert[10*i:])/(len(Ftot_vert)-10*i)
        elif 10*i+10 <= len(Ftot_vert):
            Ftot_vert_corr[i]=-np.sum(Ftot_vert[10*i:10*i+10])/10
            
    vgrf1_corr = np.zeros(len(t))
    for i in range(0,len(t)):
        if 10*i+10 > len(vgrf1):
            vgrf1_corr[i]=-np.sum(vgrf1[10*i:])/(len(vgrf1)-10*i)
        elif 10*i+10 <= len(vgrf1):
            vgrf1_corr[i]=-np.sum(vgrf1[10*i:10*i+10])/10
            
    vgrf2_corr = np.zeros(len(t))
    for i in range(0,len(t)):
        if 10*i+10 > len(vgrf2):
            vgrf2_corr[i]=-np.sum(vgrf2[10*i:])/(len(vgrf2)-10*i)
        elif 10*i+10 <= len(vgrf2):
            vgrf2_corr[i]=-np.sum(vgrf2[10*i:10*i+10])/10
        
    
    #split MoCap data into segments
    segments = SegmentationNew.segmentation(markerpos,labelsMoCap)
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
        
    
    lenMS = len(t)
    lenMC = len(segmentscorr[0])
    lenF = len(Ftot_vert_corr)
    # verifier parce que du coup coupé les valeurs Mocap pr qqes experiences
    len_corr = min(lenMS, lenMC, lenF)
    
    #SE_FL5_1
    # len_corr = 40000
    
    out_dict = {}
    
    if have_MS_data == True :
        out_dict["AlphaShank"] = AlphaShank[0: len_corr]
        out_dict["AlphaThigh"] = AlphaThigh[0: len_corr]
        out_dict["AlphaTrunk"] = AlphaTrunk[0: len_corr]
        # out_dict["EncCount"] = EncCount[0: len_corr]
        out_dict["CurrentRead"] = CurrentRead[0: len_corr]
        out_dict["CurrentSent"] = CurrentSent[0: len_corr]
        out_dict["R_leg"] = R_Leg[0: len_corr]
        out_dict["L_leg"] = L_Leg[0: len_corr]
        out_dict["Mode"] = Mode[0: len_corr]
        out_dict["Force"] = Force[0: len_corr]
        out_dict["ForceLevel"] = ForceLevel[0: len_corr]
        out_dict["GyroCThigh"] = GyroCThigh[0: len_corr]
        out_dict["GyroCShank"] = GyroCShank[0: len_corr]
        out_dict["AccelAThigh"] = AccelAThigh[0: len_corr]
        out_dict["AccelAShank"] = AccelAShank[0: len_corr]
        
        #andri
        out_dict["GyroAThigh"] = GyroAThigh[0: len_corr]
        out_dict["GyroAShank"] = GyroAShank[0: len_corr]
        out_dict["AccelBThigh"] = AccelBThigh[0: len_corr]
        out_dict["AccelBShank"] = AccelBShank[0: len_corr]
        out_dict["GyroBThigh"] = GyroBThigh[0: len_corr]
        out_dict["GyroBShank"] = GyroBShank[0: len_corr]
        out_dict["AccelCThigh"] = AccelCThigh[0: len_corr]
        out_dict["AccelCShank"] = AccelCShank[0: len_corr]
        
        out_dict["HallSensor"] = HallSensor[0: len_corr]
    
    # ignore labelsMoCap et markerpos - pas utilisé (vu que info est transféré 
    # ds segment) sauf pr la longueur pr la suite - trouver autre façon
    
    # ignore principal directions aussi 
    
    out_dict["time"] = t[0: len_corr]
    out_dict["vgrf"] = Ftot_vert_corr[0: len_corr]
    out_dict["ShankSegment"] = segmentscorr[0][0: len_corr]
    out_dict["ThighSegment"] = segmentscorr[1][0: len_corr]
    out_dict["KMAlowSegment"] = segmentscorr[2][0: len_corr]
    out_dict["KMAupSegment"] = segmentscorr[3][0: len_corr]
    out_dict["TDUSegment"] = segmentscorr[4][0: len_corr]
    out_dict["ShouldersSegment"] = segmentscorr[5][0: len_corr]
    
    out_dict["ResNormShank"] = res_norm_allseg[0][0: len_corr]
    out_dict["ResNormThigh"] = res_norm_allseg[1][0: len_corr]
    out_dict["ResNormKMAlow"] = res_norm_allseg[2][0: len_corr]
    out_dict["ResNormKMAup"] = res_norm_allseg[3][0: len_corr]
    out_dict["ResNormTDU"] = res_norm_allseg[4][0: len_corr]
    out_dict["ResNormShoulders"] = res_norm_allseg[5][0: len_corr]
    
    out_dict["vgrf1"] = vgrf1_corr[0: len_corr]
    out_dict["vgrf2"] = vgrf2_corr[0: len_corr]

   
    print("Data Processing Collect end")
    
    return out_dict, segmentscorr