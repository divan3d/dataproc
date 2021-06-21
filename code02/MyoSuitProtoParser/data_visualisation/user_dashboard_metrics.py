# -*- coding: utf-8 -*-
"""
Created on Mon March 31 2020

@author: Gleb Koginov
"""

import numpy as np
import glob as gleb
import pickle
from scipy.integrate import cumtrapz
import scipy.interpolate as interp
from scipy import stats
from scipy import signal
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
import pandas as pd 
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from itertools import groupby
from operator import itemgetter
from shutil import copy



# dateRange = [70,80]

VERBOSE = 0

# remove outliers


class dashboard_metrics():
    
    def __init__(self, inputData, legacy):
        self.inputData = inputData
        # Default Data 
        # self.inputData = inputData['TDU_DATA_INTERP']
        self.time = [s.ToMilliseconds()/1000 for s in inputData.DataTimestamp.values]#inputData['TIME_VECTOR']
        
        # Main vars #
        # Biomechanics #
        self.RTAFlt = self.inputData['RT_AFlt'].values / 1000
        self.LTAFlt = self.inputData['LT_AFlt'].values / 1000
        self.RSAFlt = self.inputData['RS_AFlt'].values / 1000
        self.LSAFlt = self.inputData['LS_AFlt'].values / 1000
        self.TrAFlt = self.inputData['Tr_AFlt'].values / 1000
        
        if not legacy:
            self.RTGyr = self.inputData['RT_GyrC'].values * 0.0175
            self.LTGyr = self.inputData['LT_GyrC'].values * 0.0175
            self.RSGyr = self.inputData['RS_GyrC'].values * 0.0175
            self.LSGyr = self.inputData['LS_GyrC'].values * 0.0175
            self.TrGyr = self.inputData['Tr_GyrC'].values * 0.0175
        else:
            self.RTGyr = self.inputData['RT_Gyr'].values * 0.0175
            self.LTGyr = self.inputData['LT_Gyr'].values * 0.0175
            self.RSGyr = self.inputData['RS_Gyr'].values * 0.0175
            self.LSGyr = self.inputData['LS_Gyr'].values * 0.0175
            self.TrGyr = self.inputData['Tr_Gyr'].values * 0.0175
            
        self.RTAccA = self.inputData['RT_AccA'].values / 4096
        self.RTAccB = self.inputData['RT_AccB'].values / 4096
        self.LTAccA = self.inputData['LT_AccA'].values / 4096
        self.LTAccB = self.inputData['LT_AccB'].values / 4096        
        self.TrAccA = self.inputData['Tr_AccA'].values / 4096
        self.TrAccB = self.inputData['Tr_AccB'].values / 4096
        self.LSAccA = self.inputData['LS_AccA'].values / 4096
        self.LSAccB = self.inputData['LS_AccB'].values / 4096
        self.RSAccA = self.inputData['RS_AccA'].values / 4096
        self.RSAccB = self.inputData['RS_AccB'].values / 4096
        
        # States #
        self.RLScon = self.inputData["R_leg"].values
        self.LLScon = self.inputData["L_leg"].values
        self.Md_F   = self.inputData['Mode'].values
        self.PatState   = self.inputData['PatState'].values
        self.Sys_inf = self.inputData['Sys_inf'].values
        # Motor Data #
        self.RM_CurR = self.inputData["RM_CurR"].values
        self.LM_CurR = self.inputData["LM_CurR"].values
        self.RLFLvl = self.inputData['RLegForce'].values
        self.LLFLvl = self.inputData['LLegForce'].values
        
        # Force Level
        self.RLegForce = self.inputData["RLegForce"].values
        self.LLegForce = self.inputData["LLegForce"].values
        
    
        # Instantiate an object ot populate with the results #
        self.results = self._resultsAll()
                
        # Nomrlize the time to 0-100% #
        self.timePercent = np.array(self.time) * 100 / self.time[-1]
        
        # Core results arrays #
        self.rlTOidx = [] # predefine right toe-off array
        self.llTOidx = [] # predefine left toe-off array
        self.rlHSidx = [] # predefine right heel-strike array
        self.llHSidx = [] # predefine left heel-strike array
            
        self.rlRMS = [] # Right motor RMS during stance phase
        self.llRMS = [] # Left motor RMS during stance phase
        
        self.rlStFLvl = np.empty(0) # Average right leg force level in stance
        self.llStFLvl = np.empty(0) # Average left leg force level in stance
        
        self.potStR = np.array([])
        self.potStL = np.array([])
        self.cadArrL = np.array([])
        self.cadArrR = np.array([])        
        
    class _resultsAll:
        # Predefine Core Arrays #
        def __init__(self):
            # Predefine Core Arrays #
            self.nRLSteps = 0
            self.nLLSteps = 0
            self.totSteps = 0  
            
            self.rlCad = 0 
            self.llCad = 0
            self.maxSpeed = 0
            self.minSpeed = 0
            self.aveSpeed = 0
            self.timeSym = 0
            self.rlStT = 0
            self.llStT = 0
            
            self.rlStepsConsidered = 0
            self.llStepsConsidered = 0
            self.llMinF = 0
            self.llMaxF = 0
            self.llAveF = 0
            self.rlAveF = 0
            self.llAveRMS = 0
        
            self.rlNumStair = 0
            self.llNumStair = 0
            self.totNumStair = 0
            
            self.totRightDistance = 0
            self.totLeftDistance = 0
            self.totDistance = 0
        
    def rejectOutliers(self, data):
        
        # Reject outer quartile dataS #
        quartiles = np.percentile(data, [25, 50, 75]) # 25%, 50% and 75% quartiles
        IQR = quartiles[2] - quartiles[0] # 
        
        return data[~((data < (quartiles[0] - 1.5 * IQR)) | (data > (quartiles[2] + 1.5 * IQR)))] # Box plot only
        # return data[~((data < (quartiles[0] - 1.5 * IQR)) | (data > (quartiles[2] + 1.5 * IQR)))] # Inner fences
    
    def idOutliers(self, data):
        # Identify the rejects #
        quartiles = np.percentile(data, [25, 50, 75]) 
        IQR = quartiles[2] - quartiles[0]
        tempArr = data[~((data < (quartiles[0] - 1.5 * IQR)) | (data > (quartiles[2] + 1.5 * IQR)))]
        
        rejected = np.array()
        for i in range(0, len(data)):
            if not data[i] in tempArr:
                rejected = np.append(i)
                
        return rejected
    
    def processWalking(self):
        
        # Filtering at 2Hz #
        b, a = scipy.signal.butter(8, 0.04, btype = "low")
        
        self.rtgFlt = scipy.signal.filtfilt(b, a, self.RTGyr) # Right thigh
        self.rsgFlt = scipy.signal.filtfilt(b, a, self.RSGyr) # Right shank
        self.ltgFlt = scipy.signal.filtfilt(b, a, self.LTGyr) # Left thigh
        self.lsgFlt = scipy.signal.filtfilt(b, a, self.LSGyr) # Left shank        
        
        self.rtgFltBU = np.copy(self.rtgFlt)
        self.rsgFltBU = np.copy(self.rsgFlt)
        self.ltgFltBU = np.copy(self.ltgFlt)
        self.lsgFltBU = np.copy(self.lsgFlt)
        
        # Find where there is no shank or thigh IMU forward movement #
        self.rtgFlt[np.where(self.rtgFlt < 20)[0]] = 0
        self.rsgFlt[np.where(self.rsgFlt < 20)[0]] = 0
        
        self.ltgFlt[np.where(self.ltgFlt < 20)[0]] = 0
        self.lsgFlt[np.where(self.lsgFlt < 20)[0]] = 0
        
        # Identify the common thigh and shank forward movement #
        self.rsgFlt[(np.where((self.rtgFlt > 0) & (self.rsgFlt > 0) & (self.Md_F > 0))[0])] = 10000        
        self.rsgFlt[np.where(self.rsgFlt < 10000)] = 0
        
        self.lsgFlt[(np.where((self.ltgFlt > 0) & (self.lsgFlt > 0) & (self.Md_F > 0))[0])] = 10000                
        self.lsgFlt[np.where(self.lsgFlt < 10000)] = 0
        
        stepsR = np.where(self.rsgFlt > 1000)[0]
        stepsL = np.where(self.lsgFlt > 1000)[0]
        peakDiffr = stepsR[np.where(np.diff(stepsR) > 1)[0] + 1] # Where separate peaks happen (right)
        peakDiffl = stepsL[np.where(np.diff(stepsL) > 1)[0] + 1] # Where separate peaks happen (left)
                
        # Right leg heel strikes and toe offs #
        for i in range(0, len(peakDiffr)):
            
            # Right leg TO #
            cnt = 0
            detFlag = 0
            while (peakDiffr[i] - cnt > 0) & (detFlag == 0):
                if self.rsgFltBU[peakDiffr[i] - cnt] < 0:
                    detFlag  = 1
                    self.rlTOidx.append(peakDiffr[i] - cnt)
                    cnt = 0
                else:
                    cnt += 1
            if (peakDiffr[i] - cnt == 0):
                self.rlTOidx.append(peakDiffr[i] - cnt)
            
            # Right leg HS #
            cnt = 0
            detFlag = 0
            while (peakDiffr[i] + cnt < len(self.rsgFlt) - 1) & (detFlag == 0):
                if self.rsgFltBU[peakDiffr[i] + cnt] < 0:
                    detFlag  = 1
                    self.rlHSidx.append(peakDiffr[i] + cnt)
                    cnt = 0
                else:
                    cnt += 1
            if (peakDiffr[i] + cnt == len(self.rsgFlt) - 1):
                self.rlHSidx.append(peakDiffr[i] + cnt)
                
        
        # Left leg heel strikes and toe offs #
        for i in range(0, len(peakDiffl)):
            
            # Left leg TO #
            cnt = 0
            detFlag = 0
            while (peakDiffl[i] - cnt > 0) & (detFlag == 0):
                if self.lsgFltBU[peakDiffl[i] - cnt] < 0:
                    detFlag  = 1
                    self.llTOidx.append(peakDiffl[i] - cnt)
                    cnt = 0
                else:
                    cnt += 1
            if (peakDiffl[i] - cnt == 0):
                self.llTOidx.append(peakDiffl[i] - cnt)
            
            # Left leg HS #
            cnt = 0
            detFlag = 0
            while (peakDiffl[i] + cnt < len(self.lsgFlt) - 1) & (detFlag == 0):
                if self.lsgFltBU[peakDiffl[i] + cnt] < 0:
                    detFlag  = 1
                    self.llHSidx.append(peakDiffl[i] + cnt)
                    cnt = 0
                else:
                    cnt += 1
            if (peakDiffl[i] + cnt == len(self.lsgFlt) - 1):
                self.llHSidx.append(peakDiffl[i] + cnt)        

        # TO and HS detected for each leg as the first and last minimum before each swing phase #
        self.true_rlTOidx = []
        self.true_rlHSidx = []
        for i in range(len(self.rlTOidx)):
            self.rsgAddedTO = self.rsgFltBU[(self.rlTOidx[i] - int(1.2 *(self.rlHSidx[i]-self.rlTOidx[i]))):self.rlTOidx[i]]
            self.rsgAddedHS = self.rsgFltBU[self.rlHSidx[i]:(self.rlHSidx[i] + int(1.2 *(self.rlHSidx[i]-self.rlTOidx[i])))]
            self.rlSwing = self.rsgFltBU[self.rlTOidx[i]:self.rlHSidx[i]]
            
            # If there is more than one peak in shank gyro for the swing phase #
            self.swingGradientDetectionR = np.where(np.diff(self.rlSwing) > 0)[0]
            if len(np.where(np.diff(self.swingGradientDetectionR)>1)[0]) > 0:
                isLast = False
                peaks = self.swingGradientDetectionR[np.where(np.diff(self.swingGradientDetectionR)>1)[0]]
                peaks = np.insert(peaks, len(peaks), self.swingGradientDetectionR[-1])
                troughs = self.swingGradientDetectionR[np.where(np.diff(self.swingGradientDetectionR)>1)[0]+1]                                
                if len(peaks) > 0: 
                    # Choose the maximum peak for each swing as a reference # 
                    maxPeak = self.rsgFltBU[self.rlTOidx[i] + peaks[0]]
                    true_minimum = troughs[0]
                    for x in range(len(peaks)):                         
                        if self.rsgFltBU[self.rlTOidx[i] + peaks[x]] > maxPeak: 
                            maxPeak = self.rsgFltBU[self.rlTOidx[i] + peaks[x]] 
                            if x < len(peaks)-1: 
                                true_minimum = troughs[x]                              
                            if x == len(peaks)-1: 
                                self.rHSgradientDetection = np.where(np.diff(self.rsgAddedHS) > 0)[0]
                                self.true_rlHSidx.append(self.rlHSidx[i] + self.rHSgradientDetection[0])  
                                isLast = True                                                                                     
                else:
                    true_minimum = self.swingGradientDetectionR[np.where(np.diff(self.swingGradientDetectionR)>1)[0]+1][0]
                if not isLast:
                    self.true_rlHSidx.append(self.rlTOidx[i] + true_minimum)                    
            # If there is only one peak than the heel strike is the first trough after 0deg/s #
            else:
                self.rHSgradientDetection = np.where(np.diff(self.rsgAddedHS) > 0)[0]
                if len(self.rHSgradientDetection > 0):
                    self.true_rlHSidx.append(self.rlHSidx[i] + self.rHSgradientDetection[0])  
                else:
                    self.true_rlHSidx.append(self.rlHSidx[i])
            self.rTOgradientDetection = np.where(np.diff(self.rsgAddedTO) > 0)[0]
            if not len(self.rTOgradientDetection): 
                self.true_rlTOidx.append(self.rlTOidx[i])
            else:
                self.rTOgradientDetection = np.insert(self.rTOgradientDetection, 0, -10)             
                rTOtoAdd = self.rTOgradientDetection[np.where(np.diff(self.rTOgradientDetection)>1)[0]+1][-1]
                self.true_rlTOidx.append(self.rlTOidx[i] - int(1.2 *(self.rlHSidx[i]-self.rlTOidx[i])) + rTOtoAdd)
        
        # TO and HS detected for each leg as the first and last minimum before each swing phase #        
        self.true_llTOidx = []
        self.true_llHSidx = []
        for i in range(len(self.llTOidx)):
            self.lsgAddedTO = self.lsgFltBU[(self.llTOidx[i] - int(1.2 *(self.llHSidx[i]-self.llTOidx[i]))):self.llTOidx[i]]
            self.lsgAddedHS = self.lsgFltBU[self.llHSidx[i]:(self.llHSidx[i] + int(1.2 * (self.llHSidx[i]-self.llTOidx[i])))]
            self.llSwing = self.lsgFltBU[self.llTOidx[i]:self.llHSidx[i]]
            
            self.swingGradientDetectionL = np.where(np.diff(self.llSwing) > 0)[0]
            if len(np.where(np.diff(self.swingGradientDetectionL)>1)[0]) > 0:
                isLast = False
                peaks = self.swingGradientDetectionL[np.where(np.diff(self.swingGradientDetectionL)>1)[0]]
                peaks = np.insert(peaks, len(peaks), self.swingGradientDetectionL[-1])
                troughs = self.swingGradientDetectionL[np.where(np.diff(self.swingGradientDetectionL)>1)[0]+1]
                if len(peaks) > 1: 
                    maxPeak = self.lsgFltBU[self.llTOidx[i] + peaks[0]]
                    true_minimum = troughs[0]
                    for x in range(len(peaks)): 
                        if self.lsgFltBU[self.llTOidx[i] + peaks[x]] > maxPeak: 
                            maxPeak = self.lsgFltBU[self.llTOidx[i] + peaks[x]]
                            if x < len(peaks)-1: true_minimum = troughs[x]                              
                            if x == len(peaks)-1: 
                                self.lHSgradientDetection = np.where(np.diff(self.lsgAddedHS) > 0)[0]
                                self.true_llHSidx.append(self.llHSidx[i] + self.lHSgradientDetection[0])  
                                isLast = True
                else:
                    true_minimum = self.swingGradientDetectionL[np.where(np.diff(self.swingGradientDetectionL)>1)[0]+1][0]
                if not isLast:
                    self.true_llHSidx.append(self.llTOidx[i] + true_minimum) 
            else:
                self.lHSgradientDetection = np.where(np.diff(self.lsgAddedHS) > 0)[0]               
                if len(self.lHSgradientDetection > 0):
                    self.true_llHSidx.append(self.llHSidx[i] + self.lHSgradientDetection[0])  
                else:
                    self.true_llHSidx.append(self.llHSidx[i])                
            self.lTOgradientDetection = np.where(np.diff(self.lsgAddedTO) > 0)[0]
            if not len(self.lTOgradientDetection): 
                self.true_llTOidx.append(self.llTOidx[i])
            else:
                self.lTOgradientDetection = np.insert(self.lTOgradientDetection, 0, -10)     
                lTOtoAdd = self.lTOgradientDetection[np.where(np.diff(self.lTOgradientDetection)>1)[0]+1][-1]
                self.true_llTOidx.append(self.llTOidx[i] - int(1.2 *(self.llHSidx[i]-self.llTOidx[i])) + lTOtoAdd)                         
        
        self.rlTOidx = self.true_rlTOidx
        self.rlHSidx = self.true_rlHSidx
        self.llTOidx = self.true_llTOidx
        self.llHSidx = self.true_llHSidx
        
        # Get rid of duplicates #
        self.rlTOidx = np.sort(np.array(list(self.rlTOidx)))
        self.rlHSidx = np.sort(np.array(list(self.rlHSidx)))
        self.llTOidx = np.sort(np.array(list(self.llTOidx)))
        self.llHSidx = np.sort(np.array(list(self.llHSidx)))
        # self.rlTOidx = np.sort(np.array(list(set(list(self.rlTOidx)))))
        # self.rlHSidx = np.sort(np.array(list(set(list(self.rlHSidx))))) 
        # self.llTOidx = np.sort(np.array(list(set(list(self.llTOidx)))))
        # self.llHSidx = np.sort(np.array(list(set(list(self.llHSidx)))))
        
        if len(self.llHSidx) > 1 and len(self.rlHSidx) > 1:
            
            self.potStR = self.rlTOidx[1::1] - self.rlHSidx[0:-1:1]
            self.potStL = self.llTOidx[1::1] - self.llHSidx[0:-1:1]               
                    
            # To be on a safer side filter outliers twice for the RMS current calculation #
            self.rlSt = self.rejectOutliers(self.rejectOutliers(self.potStR))
            self.llSt = self.rejectOutliers(self.rejectOutliers(self.potStL))            
            
            # Cumulative data #
            self.stepsCumltL = np.linspace(0, 100, len(self.llHSidx))
            self.stepIDNormL = self.llHSidx * 100 / (self.llHSidx[-1])
            self.stepsCumltR = np.linspace(0, 100, len(self.rlHSidx))
            self.stepIDNormR = self.rlHSidx * 100 / (self.rlHSidx[-1])            
            
            self.results.nRLSteps = len(self.rlHSidx)
            self.results.nLLSteps = len(self.llHSidx)
            self.results.totSteps = self.results.nRLSteps + self.results.nLLSteps
            print("Right leg level steps: {}".format(self.results.nRLSteps))
            print("Left leg level steps: {}".format(self.results.nLLSteps))
            if len(self.llHSidx) > 5 and len(self.rlHSidx) > 5:
                self.step_count_low = 0
            else:
                self.step_count_low = 1
        else:
            
            self.step_count_low = 1
            self.results.nRLSteps = 0
            self.results.nLLSteps = 0
            self.results.totSteps = 0            
        
    def processCadence(self):
        
        if not self.results.totSteps:
            self.results.rlCad = 0 
            self.results.llCad = 0
            self.results.maxSpeed = 0
            self.results.minSpeed = 0
            self.results.aveSpeed = 0
            self.results.timeSym = 0
            self.results.rlStT = 0
            self.results.llStT = 0
            
        else:
            # Time spent in stance phase #
            self.results.rlStT = np.median(self.rejectOutliers(self.potStR))/100
            self.results.llStT = np.median(self.rejectOutliers(self.potStL))/100
            
            # Cadence #       
            self.cadArrR = self.rejectOutliers(self.rejectOutliers(self.rlHSidx[1::1] - self.rlHSidx[0:-1:1])) / 100
            self.cadArrL = self.rejectOutliers(self.rejectOutliers(self.llHSidx[1::1] - self.llHSidx[0:-1:1])) / 100
            self.cadArrR = self.cadArrR[self.cadArrR > 0]
            self.cadArrL = self.cadArrL[self.cadArrL > 0]
            
            
            if len(self.cadArrL) or len(self.cadArrR):
                self.results.rlCad = np.mean(self.cadArrR) 
                self.results.llCad = np.mean(self.cadArrL) 
                            
                self.stepsCmlt = np.sort(np.hstack((self.stepsCumltL, self.stepsCumltR))[:-1]) # Cumulative distribution of the steps
                self.stepIDNorm = np.sort(np.hstack((self.stepIDNormL, self.stepIDNormR))[:-1]) # idxs of the steps normalized to 0-100% session
                             
                # Divide the session into intervals. #
                self.rlStepsInterval = np.where((np.diff(self.rlTOidx) > self.results.rlCad * 150))[0]
                self.llStepsInterval = np.where(np.diff(self.llTOidx) > self.results.llCad * 150)[0]
                
                # Concider only the intervals that include more than 3 steps #
                if self.rlStepsInterval.size > 1 and self.llStepsInterval.size > 1 and max(np.diff(self.llStepsInterval)) > 3 and max(np.diff(self.llStepsInterval)) > 3:
                    
                    if self.rlStepsInterval[0] != 0: self.rlStepsInterval = np.insert(self.rlStepsInterval,0,0)
                    if self.llStepsInterval[0] != 0: self.llStepsInterval = np.insert(self.llStepsInterval,0,0)
    
                    # Only concider intervals that include 3 or more steps #                  
                    self.LongerIntervalsStartR = self.rlStepsInterval[np.where(np.diff(self.rlStepsInterval) > 3)[0]]                    
                    self.LongerIntervalsEndR = self.rlStepsInterval[np.where(np.diff(self.rlStepsInterval) > 3)[0]+1] 
                    self.LongerIntervalsStartL = self.llStepsInterval[np.where(np.diff(self.llStepsInterval) > 3)[0]]
                    self.LongerIntervalsEndL = self.llStepsInterval[np.where(np.diff(self.llStepsInterval) > 3)[0]+1] 
                    
                    if len(self.LongerIntervalsStartR) > len(self.LongerIntervalsStartL): 
                        self.LongerIntervalsStartR = self.LongerIntervalsStartR[:len(self.LongerIntervalsStartL)]
                        self.LongerIntervalsEndR = self.LongerIntervalsEndR[:len(self.LongerIntervalsEndL)]
                    if len(self.LongerIntervalsStartR) < len(self.LongerIntervalsStartL): 
                        self.LongerIntervalsStartL = self.LongerIntervalsStartL[:len(self.LongerIntervalsStartR)]
                        self.LongerIntervalsEndL = self.LongerIntervalsEndL[:len(self.LongerIntervalsEndR)]
                    
                    # Look at the length of each interval # 
                    DifferenceIntervalsR = np.diff([self.LongerIntervalsStartR, self.LongerIntervalsEndR], axis=0)[0]
                    DifferenceIntervalsL = np.diff([self.LongerIntervalsStartL, self.LongerIntervalsEndL], axis=0)[0]                
                    
                    # Calculate the cadence for each step of the first interval #
                    CadenceOfFirstIntR = self.rlHSidx[(self.LongerIntervalsStartR[0] + 2):(self.LongerIntervalsEndR[0])] - self.rlHSidx[(self.LongerIntervalsStartR[0] + 1):(self.LongerIntervalsEndR[0]-1)]                    
                    CadenceOfFirstIntL = self.llHSidx[(self.LongerIntervalsStartL[0] + 2):(self.LongerIntervalsEndL[0])] - self.llHSidx[(self.LongerIntervalsStartL[0] + 1):(self.LongerIntervalsEndL[0]-1)]     
                    
                    self.results.timeSym = np.min([np.mean(CadenceOfFirstIntR), np.mean(CadenceOfFirstIntL)])/np.max([np.mean(CadenceOfFirstIntR), np.mean(CadenceOfFirstIntL)])
                    
                else:
                    self.results.timeSym = (np.min([self.results.rlCad, self.results.llCad]) / np.max([self.results.rlCad, self.results.llCad]))    
                
                # Extract the walking speed metrics #                
                min_cadenceR = np.min(self.cadArrR)
                mean_cadenceR = np.mean(self.cadArrR)
                max_cadenceR = np.max(self.cadArrR)
                min_cadenceL = np.min(self.cadArrL)
                mean_cadenceL = np.mean(self.cadArrL)
                max_cadenceL = np.max(self.cadArrL)
                                
                # Max, min, and average cadence including both legs. #
                self.results.maxSpeed = int(60/(min_cadenceR * (min_cadenceR > 0)) + 60/(min_cadenceL * (min_cadenceL > 0)))
                self.results.minSpeed = int(60/(max_cadenceR * (max_cadenceR > 0)) + 60/(max_cadenceL * (max_cadenceL > 0)))
                self.results.aveSpeed = int(60/(mean_cadenceR * (mean_cadenceR > 0)) + 60/(mean_cadenceL * (mean_cadenceL > 0)))                                
            else:
                self.results.rlCad = 0 
                self.results.llCad = 0
                self.results.maxSpeed = 0
                self.results.minSpeed = 0
                self.results.aveSpeed = 0
                self.results.timeSym = 0
                
        print("Right leg ave cadence time: {}(s)".format(self.results.rlCad))
        print("Left leg ave cadence time: {}(s)".format(self.results.llCad))
        print("Max speed: {}(step/min)".format(self.results.maxSpeed))
        print("Min speed: {}(step/min)".format(self.results.minSpeed))
        print("Ave speed: {}(step/min)".format(self.results.aveSpeed))
        print("Right leg ave stance time: {}(s)".format(self.results.rlStT))
        print("Left leg ave stance time: {}(s)".format(self.results.llStT))
        print("Walking time symmetry: {}(%)".format(self.results.timeSym * 100))
        
        
    def processMotorData(self):
        # Performance of the motor during the stance phases #
        self.results.rlStepsConsidered = 0
        self.results.llStepsConsidered = 0
        # Right Motor #
        if self.potStR.size:
            for i in range(0, len(self.potStR)):
                if self.potStR[i] not in self.rlSt:
                    continue
                else:
                    self.rlRMS.append(np.sqrt(np.mean(self.RM_CurR[self.rlHSidx[i]:self.rlTOidx[i+1]] ** 2)))
                    tmpFLvls = self.RLFLvl[self.rlHSidx[i]:self.rlTOidx[i+1]]
                    tmpFLvls = tmpFLvls[tmpFLvls > -1] # reject the steps done before logging engaged
                    if len(tmpFLvls):
                        # Stance phase is all in Assistance State #
                        if all(x == 9 for x in self.Sys_inf[self.rlHSidx[i]:self.rlTOidx[i+1]]):
                            self.rlStFLvl = np.append(self.rlStFLvl, np.mean(tmpFLvls)) 
                            self.results.rlStepsConsidered += 1
                        # Stance phase is partly in Transparency and partly in Assistance State #
                        elif not all(x == 8 for x in self.Sys_inf[self.rlHSidx[i]:self.rlTOidx[i+1]]):                         
                            AssistState = np.where(self.Sys_inf[self.rlHSidx[i]:self.rlTOidx[i+1]] == 9)[0]
                            TranspState = np.where(self.Sys_inf[self.rlHSidx[i]:self.rlTOidx[i+1]] == 8)[0]                                                        
                            AssistMean = np.mean(self.RLFLvl[self.rlHSidx[i] + AssistState])

                            weighted_avg = np.average([AssistMean, 0.0], weights = [len(AssistState), len(TranspState)])
                            self.rlStFLvl = np.append(self.rlStFLvl, weighted_avg)
                            self.results.rlStepsConsidered += 1
                        # Stance phase is all in Transparency State #
                        else:
                            self.rlStFLvl = np.append(self.rlStFLvl, 0.0)
                            self.results.rlStepsConsidered += 1
                            
            self.results.rlAveF = np.round(np.mean(self.rlStFLvl)) # Average force in stance (level)
            
            if max(self.rlStFLvl) > 0:
                self.results.rlMinF = np.min(self.rlStFLvl[self.rlStFLvl > 0])
                self.results.rlMaxF = np.max(self.rlStFLvl[self.rlStFLvl > 0])
                self.results.rlAveRMS = np.round(np.mean(self.rlRMS),2) # Average RMS in stance
            else:
                self.results.rlMinF = 0
                self.results.rlMaxF = 0
                self.results.rlAveRMS = 0
                
        else:
            if all(x < 9 for x in self.Sys_inf):
                self.rlStFLvl = 0.0
            else:
                self.rlStFLvl = np.round(np.mean(self.RLFLvl))
            self.results.rlAveF = np.round(np.mean(self.rlStFLvl)) # Average force in stance (level)
            self.results.rlMinF = np.min(self.rlStFLvl)
            self.results.rlMaxF = np.max(self.rlStFLvl)            
            self.results.rlAveRMS = np.sqrt(np.mean(self.RM_CurR ** 2)) # Average RMS in stance
        
        # Left Motor
        if self.potStL.size:
            for i in range(0, len(self.potStL)):
                if self.potStL[i] not in self.llSt:
                    continue
                else:
                    self.llRMS.append(np.sqrt(np.mean(self.LM_CurR[self.llHSidx[i]:self.llTOidx[i+1]] ** 2)))
                    tmpFLvls = self.LLFLvl[self.llHSidx[i]:self.llTOidx[i+1]]
                    tmpFLvls = tmpFLvls[tmpFLvls > -1] # reject the steps done before logging engaged
                    if len(tmpFLvls):
                        # Stance phase is all in Assistance State #
                        if all(x == 9 for x in self.Sys_inf[self.llHSidx[i]:self.llTOidx[i+1]]):
                            self.llStFLvl = np.append(self.llStFLvl, np.mean(tmpFLvls)) 
                            self.results.llStepsConsidered += 1                            
                        # Stance phase is partly in Transparency and partly in Assistance State #
                        elif not all(x == 8 for x in self.Sys_inf[self.llHSidx[i]:self.llTOidx[i+1]]):                         
                            AssistState = np.where(self.Sys_inf[self.llHSidx[i]:self.llTOidx[i+1]] == 9)[0]
                            TranspState = np.where(self.Sys_inf[self.llHSidx[i]:self.llTOidx[i+1]] == 8)[0]                                                                                    
                            AssistMean = np.mean(self.LLFLvl[self.llHSidx[i] + AssistState])
                            
                            weighted_avg = np.average([AssistMean, 0.0], weights = [len(AssistState), len(TranspState)])                            
                            self.llStFLvl = np.append(self.llStFLvl, weighted_avg)                            
                            self.results.llStepsConsidered += 1
                        # Stance phase is all in Transparency State #
                        else:
                            self.llStFLvl = np.append(self.llStFLvl, 0.0)
                            self.results.llStepsConsidered += 1                                                
            
            self.results.llAveF = np.round(np.mean(self.llStFLvl)) # Average force in stance (level)
            if max(self.llStFLvl) > 0: 
                self.results.llMinF = np.min(self.llStFLvl[self.llStFLvl > 0])
                self.results.llMaxF = np.max(self.llStFLvl[self.llStFLvl > 0])
                self.results.llAveRMS = np.round(np.mean(self.llRMS),2) # Average RMS in stance
            else:
                self.results.llMinF = 0
                self.results.llMaxF = 0
                self.results.llAveRMS = 0
                
        else:
            if all(x < 9 for x in self.Sys_inf):
                self.llStFLvl = 0.0
            else:
                self.llStFLvl = np.round(np.mean(self.LLFLvl))
            self.results.llAveF = np.round(np.mean(self.llStFLvl)) # Average force in stance (level)
            self.results.llMinF = np.min(self.llStFLvl)
            self.results.llMaxF = np.max(self.llStFLvl)
            self.results.llAveRMS = np.sqrt(np.mean(self.LM_CurR ** 2)) # Average RMS in stance
                                                    
        print('Average force level in stance - right leg: {}'.format(self.results.rlAveF))
        print('Number of steps considered - right leg: {}'.format(self.results.rlStepsConsidered))
        print('Average force level in stance - left leg: {}'.format(self.results.llAveF))                   
        print('Number of steps considered - left leg: {}'.format(self.results.llStepsConsidered))
        
        
    # Detect sit-to-stand instances #
    def processSTS(self):   
        self.STSThresh   = 150 # Minimum angles required for STS detection                              
        self.stsIds = np.where((self.RTAFlt >= self.STSThresh) & (self.LTAFlt >= self.STSThresh))[0]                               
                        
        notActiveCounter = 0
        if len(self.rlHSidx) and len(self.rlTOidx): lengthOfFirstSwing = self.rlHSidx[0] - self.rlTOidx[0]        
        else: lengthOfFirstSwing = 100
        
        for i in range(0,(len(self.time)-2*lengthOfFirstSwing)):
            # Inactive within a certain interval
            if max(abs(self.rtgFltBU[i:(i+2*lengthOfFirstSwing)])) < 10 and max(abs(self.ltgFltBU[i:(i+2*lengthOfFirstSwing)])) < 10 and max(abs(self.rsgFltBU[i:(i+2*lengthOfFirstSwing)])) < 10 and max(abs(self.lsgFltBU[i:(i+2*lengthOfFirstSwing)])) < 10:            
                notActiveCounter += 1   
          
        self.results.tNotSit = np.round(self.time[-1]/ 1000/ 60 - notActiveCounter / 100 / 60, 2)
        self.results.notSitFract = int(100 * (1-notActiveCounter/(len(self.time)-2*lengthOfFirstSwing)))
        
        print("Total time active: {}(min)".format(self.results.tNotSit))
        print("Percentage of session active: {}(%)".format(self.results.notSitFract))

        
        # If there are no sitting transfers
        if not self.stsIds.size or len(self.stsIds) == len(self.LTAFlt):            
            self.results.numSTS = 0
            self.results.numSquats = 0     
            
            print("Number of sitting transfers: {}".format(self.results.numSTS))           
            print("Number of squats: {}".format(self.results.numSquats))
            return  
        
        self.stsIds  = np.insert(self.stsIds, len(self.stsIds), self.stsIds[len(self.stsIds)-1]+100)
        self.stsIds  = np.insert(self.stsIds, 0, -100) 
        self.standUp = self.stsIds[np.where(np.diff(self.stsIds)>5)[0]]                       
        self.sitDown = self.stsIds[np.where(np.diff(self.stsIds)>5)[0]+1]
        
        self.standUp = self.standUp[1:len(self.standUp)] # remove -100 value
        self.sitDown = self.sitDown[0:len(self.sitDown)-1] # remove self.stsIds[len(self.stsIds)-1]+100 value
        
        # STS in concentric or transparent and squats in isometric or transparent       
        self.stsDiffIndex   = list()
        self.squatDiffIndex = list()
        self.numberOfSTS    = 0
        self.numberOfSquats = 0
        self.squatTime = 0
        diffIndexSTS   = [0]
        for i in range(min(len(self.standUp),len(self.sitDown))):
            if self.standUp[i] > self.sitDown[i]:
                if self.Sys_inf[self.sitDown[i]] == 9 and self.Sys_inf[self.standUp[i]] == 9:
                    if self.Md_F[self.sitDown[i]] == 2 or self.Md_F[self.standUp[i]] == 2:                
                        self.numberOfSquats += 1 
                        self.squatTime += self.standUp[i] - self.sitDown[i]
                    elif self.Md_F[self.sitDown[i]] == 1 and self.Md_F[self.standUp[i]] == 1:                
                        self.numberOfSTS += 1      
                    else:
                        diffIndexSTS = [self.standUp[i] - self.sitDown[i]]
                elif self.Sys_inf[self.sitDown[i]] == 8 or self.Sys_inf[self.standUp[i]] == 8:
                    diffIndexSTS = [self.standUp[i] - self.sitDown[i]]
            else:
                if self.Sys_inf[self.sitDown[i]] == 9 and self.Sys_inf[self.standUp[i]] == 9:
                    if self.Md_F[self.sitDown[i]] == 2 or self.Md_F[self.standUp[i]] == 2:
                        self.numberOfSquats += 1   
                        self.squatTime += self.sitDown[i] - self.standUp[i]
                    elif self.Md_F[self.sitDown[i]] == 1 and self.Md_F[self.standUp[i]] == 1:
                        self.numberOfSTS += 1
                    else:
                        diffIndexSTS   = [self.sitDown[i] - self.standUp[i]]
                elif self.Sys_inf[self.sitDown[i]] == 8 or self.Sys_inf[self.standUp[i]] == 8:
                    diffIndexSTS   = [self.sitDown[i] - self.standUp[i]]
            self.stsDiffIndex.extend(diffIndexSTS)
            diffIndexSTS   = [0]
        
        
        isSitToStand = False        
        for i in range(len(self.stsDiffIndex)):
            if self.stsDiffIndex[i] != 0:
                self.TrGyr_lowSpeed = np.where(self.TrGyr[self.sitDown[i]:self.sitDown[i]+self.stsDiffIndex[i]] < 10)[0]            
                self.ThighGyr_lowspeed = np.where((abs(self.LTGyr[self.sitDown[i]:self.sitDown[i]+self.stsDiffIndex[i]]) < 10)
                                                  & (abs(self.RTGyr[self.sitDown[i]:self.sitDown[i]+self.stsDiffIndex[i]]) < 10))[0]         
                for j in range(self.stsDiffIndex[i]): 
                    # Trunk angle higher than 80 when doing STS
                    if self.TrAFlt[self.sitDown[i]+j] > 80 and ~isSitToStand:
                        isSitToStand = True 
                # Trunk Gyro seems to be lower than 10deg/s less than 86% of the sitting sequence
                # or Thigh Gyros seem to be lower than 10deg/s more than 82% of the sitting sequence.
                # This is based on the fact that you might stay longer in a sitting position on a chair than in a squat
                # and that you go down slower when doing a squat.
                if (len(self.TrGyr_lowSpeed)/self.stsDiffIndex[i] < 0.86 or len(self.ThighGyr_lowspeed)/self.stsDiffIndex[i] > 0.82 ) and isSitToStand:
                    self.numberOfSTS += 1  
                else: 
                    self.numberOfSquats += 1
                    self.squatTime += self.stsDiffIndex[i]
                isSitToStand = False                
                
        self.results.numSTS = self.numberOfSTS
        self.results.numSquats = self.numberOfSquats                
        
        print("Number of sitting transfers: {}".format(self.results.numSTS))
        print("Number of squats: {}".format(self.results.numSquats))  
        
    def processStairs(self, myosuit_calibration = [], stair_calibration = []):
                                
        if len(self.rlTOidx) <= 5 or len(self.llTOidx) <= 5  :
            self.results.rlNumStair = 0
            self.results.llNumStair = 0
            self.results.totNumStair = 0
            
            print('Number of stairs - right leg: {}'.format(self.results.rlNumStair))
            print('Number of stairs - left leg: {}'.format(self.results.llNumStair))
            
            return
        
        elif not myosuit_calibration:
            # Prepare the empty arrays to append #
            self.rtMax = np.empty((6,0))
            self.ltMax = np.empty((6,0))
            # Find mean shank angle and max shank gyro values during the first 6 steps #
            for i in range(0,6):
                RT_segment = self.RTAFlt[self.rlTOidx[i]:self.rlHSidx[i]]
                LT_segment = self.LTAFlt[self.llTOidx[i]:self.llHSidx[i]] 
                self.rtMax = np.append(self.rtMax, np.max(RT_segment))
                self.ltMax = np.append(self.ltMax, np.max(LT_segment))
                
            self.rtStepThresh = np.mean(self.rtMax)
            self.ltStepThresh = np.mean(self.ltMax)  
            
        
            
        else:
            # Correct the data wrapping around for wrong data type in the old SQL database #
            if myosuit_calibration[0] < 0:
                myosuit_calibration[0] = 32767+(32767-myosuit_calibration[0])
                
            if myosuit_calibration[1] < 0:
                myosuit_calibration[0] = 32767+(32767-myosuit_calibration[1])
                
            self.rtStepThresh = (myosuit_calibration[0] + 90)
            self.ltStepThresh = (myosuit_calibration[1] + 90)            
            
        if not stair_calibration:
            # Prepare the empty arrays to append #
            self.meanRShankAngle = np.empty((0,6))
            self.meanLShankAngle = np.empty((0,6))
            self.maxRShankGyro = np.empty((0,6))
            self.maxLShankGyro = np.empty((0,6))
            self.rkMax = np.empty((0,6))
            self.lkMax = np.empty((0,6)) 
            
            # Find mean shank angle and max shank gyro values during the first 6 steps #
            for i in range(0,6):
                
                # SHANK ANGLE #
                self.meanRShankAngle = np.append(self.meanRShankAngle, np.mean(self.RSAFlt[self.rlTOidx[i]:self.rlHSidx[i]]))
                self.meanLShankAngle = np.append(self.meanLShankAngle, np.mean(self.LSAFlt[self.llTOidx[i]:self.llHSidx[i]]))
                
                # SHANK GYRO #
                self.maxRShankGyro = np.append(self.maxRShankGyro, np.max(self.rsgFltBU[self.rlTOidx[i]:self.rlHSidx[i]]))
                self.maxLShankGyro = np.append(self.maxLShankGyro, np.max(self.lsgFltBU[self.llTOidx[i]:self.llHSidx[i]]))
                
                # KNEE SEGMENTS #
                RT_segment = self.RTAFlt[self.rlTOidx[i]:self.rlHSidx[i]]
                RS_segment = self.RSAFlt[self.rlTOidx[i]:self.rlHSidx[i]]         
                LT_segment = self.LTAFlt[self.llTOidx[i]:self.llHSidx[i]]
                LS_segment = self.LSAFlt[self.llTOidx[i]:self.llHSidx[i]]
                self.rkMax = np.append(self.rkMax, np.max(RT_segment - RS_segment))
                self.lkMax = np.append(self.lkMax, np.max(LT_segment - LS_segment))
                
            self.rsStepThresh = np.mean(self.meanRShankAngle)
            self.lsStepThresh = np.mean(self.meanLShankAngle)    
            self.rgStepThresh = np.mean(self.maxRShankGyro)
            self.lgStepThresh = np.mean(self.maxLShankGyro) 
            self.rKnee_threshold = np.mean(self.rkMax)       
            self.lKnee_threshold = np.mean(self.lkMax)
            
        else:
            self.rsStepThresh = stair_calibration[0]
            self.lsStepThresh = stair_calibration[1]
            self.rgStepThresh = stair_calibration[2]
            self.lgStepThresh = stair_calibration[3]
            self.rKnee_threshold = stair_calibration[4]
            self.lKnee_threshold = stair_calibration[5]
            
        
        # Take into account all steps. #
        self.rlPotStair = np.empty((len(self.rlTOidx),0))
        self.llPotStair = np.empty((len(self.llTOidx),0))
        for i in range(min(len(self.rlHSidx),len(self.rlTOidx))):
            self.rlPotStair = np.append(self.rlPotStair, np.arange(self.rlTOidx[i],self.rlHSidx[i]+1))                                     
        for i in range(min(len(self.llTOidx),len(self.llHSidx))):
            self.llPotStair = np.append(self.llPotStair, np.arange(self.llTOidx[i],self.llHSidx[i]+1))                                     
        self.rlPotStair = self.rlPotStair.astype(np.int64)
        self.llPotStair = self.llPotStair.astype(np.int64)                
        
        # Assemble the idx sequences into separate arrays where breaks happen #
        self.rlStep = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(self.rlPotStair), lambda x: x[0]-x[1])]
        self.rlStep = [x for x in self.rlStep if len(x) > 50]
        self.llStep = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(self.llPotStair), lambda x: x[0]-x[1])]
        self.llStep = [x for x in self.llStep if len(x) > 50]
        
        # Only concider the steps which exceed the relevant thigh angle. #
        self.rlStairSec = []
        self.llStairSec = []        
        for i in range(len(self.llStep)):
            if max(self.LTAFlt[self.llStep[i]]) > self.ltStepThresh: 
                self.llStairSec.append(self.llStep[i])
        for i in range(len(self.rlStep)):
            if max(self.RTAFlt[self.rlStep[i]]) > self.rtStepThresh: 
                self.rlStairSec.append(self.rlStep[i])          
        
        self.rKneeAngle = self.RTAFlt - self.RSAFlt
        self.lKneeAngle = self.LTAFlt - self.LSAFlt
                      
        # Check the potential stairs that are not sitting transitions #
        rlCount = 0 # Right leg counter
        llCount = 0 # Left leg counter     
        
        # Right leg loop #
        for i in range(0, len(self.rlStairSec)):
            
            # Check that the range does not start with 0 #
            if self.rlStairSec[i][0] == 0:
                rngMin = [0]
            else:
                rngMin = np.arange(max(0, self.rlStairSec[i][0] - 50), self.rlStairSec[i][0], 1)
            
            # Check that the range does not end with len - 1 == [-1]
            if self.rlStairSec[i][-1] == len(self.RTAFlt) - 1:
                rngMax = [len(self.RTAFlt) - 1]
            else:
                rngMax = np.arange(self.rlStairSec[i][-1]+1, min(len(self.RTAFlt)-1, self.rlStairSec[i][-1] + 51),1)
            
            full_range = np.concatenate((rngMin, self.rlStairSec[i], rngMax))
            
            # Check that the person is not sitting in the range -500[ms]:+500[ms] of the potential range (to ensure that this is not a sitting transfer) #
            if (max(self.RTAFlt[rngMin]) > self.STSThresh) and (max(self.LTAFlt[rngMin]) > self.STSThresh):
                continue
            elif (max(self.RTAFlt[rngMax]) > self.STSThresh) and (max(self.LTAFlt[rngMax]) > self.STSThresh):
                continue
            # Check that within this range the correlation coefficient of the two signals is low (not a slow sitting transition) #
            elif pearsonr(self.RTAFlt[full_range], self.LTAFlt[full_range])[0] > 0.75:
                continue
            # Check that the users knee angle exceeds the knee threshold #
            elif np.max(self.RTAFlt[self.rlStairSec[i]] - self.RSAFlt[self.rlStairSec[i]]) < self.rKnee_threshold:                                                
                continue
            # Check if the leg goes above 145deg thigh angle, which could be a hip flexion exercise #
            elif not all(x < 145 for x in self.RTAFlt[self.rlStairSec[i]]):                
                continue
            # The shank angle should not exceed the mean shank angle for regular walking #
            elif not all(x < 1.2 * self.rsStepThresh for x in self.RSAFlt[self.rlStairSec[i]]):                                                
                continue
            # The shank gyro is lower for stair ascent than for level ground walking #
            elif max(self.rsgFltBU[self.rlStairSec[i]]) < 0.67 * self.rgStepThresh:                              
                rlCount += 1
        # Left leg loop #
        for i in range(0, len(self.llStairSec)):
            
            # Check that the range does not start with 0 #
            if self.llStairSec[i][0] == 0:
                rngMin = [0]
            else:
                rngMin = np.arange(max(0, self.llStairSec[i][0] - 50), self.llStairSec[i][0], 1)
            
            # Check that the range does not end with len - 1 == [-1]
            if self.llStairSec[i][-1] == len(self.LTAFlt) - 1:
                rngMax = [len(self.LTAFlt) - 1]
            else:
                rngMax = np.arange(self.llStairSec[i][-1], min(len(self.LTAFlt)-1, self.llStairSec[i][-1] + 51),1)
                
            full_range = np.concatenate((rngMin, self.llStairSec[i], rngMax))
            
            # Check that the person is not sitting in the range -500[ms]:+500[ms] of the potential range (to ensure that this is not a sitting transfer) #
            if (max(self.RTAFlt[rngMin]) > self.STSThresh) and (max(self.LTAFlt[rngMin]) > self.STSThresh):
                continue
            elif(max(self.RTAFlt[rngMax]) > self.STSThresh) and (max(self.LTAFlt[rngMax]) > self.STSThresh):
                continue
            # Check that within this range the correlation coefficient of the two signals is low (not a slow sitting transition) #
            elif pearsonr(self.RTAFlt[full_range], self.LTAFlt[full_range])[0] > 0.75:
                continue
            # Check that the users knee angle exceeds the knee threshold #                 
            elif np.max(self.LTAFlt[self.llStairSec[i]] - self.LSAFlt[self.llStairSec[i]]) < self.lKnee_threshold:  
                continue
            # Check if the leg goes above 145deg thigh angle
            elif not all(x < 145 for x in self.LTAFlt[self.llStairSec[i]]):                
                continue      
            # The shank angle should not exceed the mean shank angle for regular walking #
            elif not all(x < 1.2 * self.lsStepThresh for x in self.LSAFlt[self.llStairSec[i]]):
                continue    
            # # The shank gyro is lower for stair ascent than for level ground walking #
            elif max(self.lsgFltBU[self.llStairSec[i]]) < 0.67 * self.lgStepThresh:
                llCount += 1    
        
        self.results.rlNumStair = rlCount
        self.results.llNumStair = llCount
        self.results.totNumStair = rlCount + llCount
        
        print('Number of stairs - right leg: {}'.format(self.results.rlNumStair))
        print('Number of stairs - left leg: {}'.format(self.results.llNumStair))                
        
    def processIntervals(self):
        if self.cadArrL.size and self.cadArrR.size:
            self.rlIntervals = np.where(np.diff(self.rlTOidx) > self.results.rlCad * 300)[0]
            self.llIntervals = np.where(np.diff(self.llTOidx) > self.results.llCad * 300)[0]
            
            self.rlIntervalsEnd = np.insert(self.rlIntervals, len(self.rlIntervals), len(self.rlTOidx)-1)
            self.rlIntervalsStart = np.insert(self.rlIntervals+1, 0, 0)
            self.llIntervalsEnd = np.insert(self.llIntervals, len(self.llIntervals), len(self.llTOidx)-1)
            self.llIntervalsStart = np.insert(self.llIntervals+1, 0, 0)
            
            
            self.rlWalkingIntervals = self.rlIntervalsEnd-self.rlIntervalsStart
            self.llWalkingIntervals = self.llIntervalsEnd-self.llIntervalsStart
            
            
            self.rlMaxInter = np.max(self.rlWalkingIntervals)
            self.llMaxInter = np.max(self.llWalkingIntervals)
            

            self.results.rlMaxStepsInt = self.rlMaxInter
            self.results.llMaxStepsInt = self.llMaxInter
            
            if self.results.rlMaxStepsInt > self.results.llMaxStepsInt:
                self.results.maxTimeInt = max((self.rlTOidx[self.rlIntervalsEnd] - self.rlTOidx[self.rlIntervalsStart] )/ 100)
                
            else:
                self.results.maxTimeInt = max((self.llTOidx[self.llIntervalsEnd] - self.llTOidx[self.llIntervalsStart] )/ 100)
                
            self.results.maxStepInt = 2 * np.max([self.results.rlMaxStepsInt, self.results.llMaxStepsInt])
        else:
            self.results.rlMaxStepsInt = 0
            self.results.llMaxStepsInt = 0
            self.results.maxTimeInt = 0
            self.results.maxStepInt = 0
    
    # def myoScore(self):
    #     if self.results.llCad > 0 or self.results.rlCad > 0:
    #         self.results.score = (self.results.totSteps + 20*self.results.numSTS) / (np.mean([self.results.llAveRMS,self.results.rlAveRMS])/5000)
    #         self.results.score = self.results.score * (5/np.mean([self.results.llCad, self.results.rlCad]))
    #     else:
    #         self.results.score = 0

    def processDistance(self):
               
        self.lowGyro = np.where((abs(self.rsgFltBU) < 1) & (abs(self.rtgFltBU) < 1) & (abs(self.lsgFltBU) < 1) & (abs(self.ltgFltBU) < 1))[0]    
        if len(self.lowGyro) > 0:
            self.rsgFltBU_MeanError = abs(np.mean(self.rsgFltBU[self.lowGyro]))
            self.lsgFltBU_MeanError = abs(np.mean(self.lsgFltBU[self.lowGyro]))
            self.rtgFltBU_MeanError = abs(np.mean(self.rtgFltBU[self.lowGyro]))
            self.ltgFltBU_MeanError = abs(np.mean(self.ltgFltBU[self.lowGyro]))
        else:
            self.rsgFltBU_MeanError = 0.08550050783441493
            self.lsgFltBU_MeanError = 0.04813070370261642
            self.rtgFltBU_MeanError = 0.3780180898704917
            self.ltgFltBU_MeanError = 0.5504482328532209
        
        l1 = 0.616
        l2 = 0.504
        sumRightLength = 0
        for i in range(0,len(self.rlTOidx)):                    
                         
            self.theta1 = cumtrapz(self.rtgFltBU[(self.rlTOidx[i]):(self.rlHSidx[i])] - self.rtgFltBU_MeanError, dx=1/100)                        
            self.theta2 = cumtrapz(self.rsgFltBU[(self.rlTOidx[i]):(self.rlHSidx[i])] - self.rsgFltBU_MeanError, dx=1/100)
        
            self.theta1L = cumtrapz(self.ltgFltBU[(self.rlTOidx[i]):(self.rlHSidx[i])] - self.ltgFltBU_MeanError, dx=1/100)                                                                       
            self.theta2L = cumtrapz(self.lsgFltBU[(self.rlTOidx[i]):(self.rlHSidx[i])] - self.lsgFltBU_MeanError, dx=1/100)                                                                       
            
            self.theta1 = self.theta1 - self.theta1[0]
            self.theta2 = self.theta2 - self.theta2[0]
            self.theta1L = self.theta1L - self.theta1L[0]
            self.theta2L = self.theta2L - self.theta2L[0]
            
            startStepX = l1*np.sin(self.theta1L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta1L[0]*np.pi/180) + l2*np.sin(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[0]*np.pi/180) + l1*np.sin(self.theta1[0]*np.pi/180) + l2*np.sin(self.theta2[0]*np.pi/180 - self.theta1[0]*np.pi/180)
            startStepY = l1*np.cos(self.theta1L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta1L[0]*np.pi/180) + l2*np.cos(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[0]*np.pi/180) - (l1*np.cos(self.theta1[0]*np.pi/180) + l2*np.cos(self.theta2[0]*np.pi/180 - self.theta1[0]*np.pi/180))
            
            endStepX = l1*np.sin(self.theta1L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta1L[len(self.theta1)-1]*np.pi/180) + l2*np.sin(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[len(self.theta1)-1]*np.pi/180) + l1*np.sin(self.theta1[len(self.theta1)-1]*np.pi/180) + l2*np.sin(self.theta2[len(self.theta1)-1]*np.pi/180 - self.theta1[len(self.theta1)-1]*np.pi/180)
            endStepY = l1*np.cos(self.theta1L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta1L[len(self.theta1)-1]*np.pi/180) + l2*np.cos(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[len(self.theta1)-1]*np.pi/180) - (l1*np.cos(self.theta1[len(self.theta1)-1]*np.pi/180) + l2*np.cos(self.theta2[len(self.theta1)-1]*np.pi/180 - self.theta1[len(self.theta1)-1]*np.pi/180))
            sumRightLength += abs(startStepX) + abs(endStepX)
            # print('x value: {}'.format(abs(startStepX) + abs(endStepX)))
            # print('y value: {}'.format(abs(abs(endStepY) - abs(startStepY))))
            
            ## Stick figure plot
            # if i==21:  
            #     plt.figure()
            #     for j in range(0,len(self.theta1)):   
            #         # toAddX = l2*np.sin(-self.theta2L[j]*np.pi/180+self.theta1L[j]*np.pi/180) + l1*np.sin(-self.theta1L[j]*np.pi/180)                    
            #         toAddX = l1*np.sin(self.theta1L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta1L[j]*np.pi/180) + l2*np.sin(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[j]*np.pi/180)#*(2-((len(self.theta1))-j)/(len(self.theta1)))
            #         # toAddY = l2*np.cos(-self.theta2L[j]*np.pi/180+self.theta1L[j]*np.pi/180) + l1*np.cos(-self.theta1L[j]*np.pi/180)
            #         toAddY = l1*np.cos(self.theta1L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta1L[j]*np.pi/180) + l2*np.cos(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[j]*np.pi/180)
                    
            #         x1L = [0,l2*np.sin(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[j]*np.pi/180)]
            #         y1L = [0,l2*np.cos(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[j]*np.pi/180)]
                    
            #         x2L = [l2*np.sin(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[j]*np.pi/180), toAddX]
            #         y2L = [l2*np.cos(self.theta2L[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2L[j]*np.pi/180), toAddY]
                    
            #         x1 = [toAddX, toAddX + l1*np.sin(self.theta1[j]*np.pi/180)]
            #         y1 = [toAddY, toAddY - l1*np.cos(self.theta1[j]*np.pi/180)]
                    
            #         x2 = [toAddX + l1*np.sin(self.theta1[j]*np.pi/180), toAddX + l1*np.sin(self.theta1[j]*np.pi/180) + l2*np.sin(self.theta2[j]*np.pi/180-self.theta1[j]*np.pi/180)]
            #         y2 = [toAddY - l1*np.cos(self.theta1[j]*np.pi/180), toAddY - (l1*np.cos(self.theta1[j]*np.pi/180) + l2*np.cos(self.theta2[j]*np.pi/180-self.theta1[j]*np.pi/180))]
                    
            #         plt.xlim(-(l1+l2)/2, (l1+l2)/2 + 0.5)
            #         plt.ylim(-0.25, (l1+l2)+0.25)
                    
            #         plt.plot(x1, y1, marker = 'o')
            #         plt.plot(x2, y2, marker = 'o')                    
                    
            #         # plt.plot(x1L, y1L, marker = 'o')
            #         # plt.plot(x2L, y2L, marker = 'o')                                        
                    
            #         plt.pause((self.rlHSidx[i] - self.rlTOidx[i])/100/len(self.theta1))
            #         plt.show()                            
        
        sumLeftLength = 0
        for i in range(0,len(self.llTOidx)):                    
                                                
            self.theta1 = cumtrapz(self.ltgFltBU[(self.llTOidx[i]):(self.llHSidx[i])] - self.ltgFltBU_MeanError, dx=1/100)                        
            self.theta2 = cumtrapz(self.lsgFltBU[(self.llTOidx[i]):(self.llHSidx[i])] - self.lsgFltBU_MeanError, dx=1/100)
           
            self.theta1R = cumtrapz(self.rtgFltBU[(self.llTOidx[i]):(self.llHSidx[i])] - self.rtgFltBU_MeanError, dx=1/100)                                                                       
            self.theta2R = cumtrapz(self.rsgFltBU[(self.llTOidx[i]):(self.llHSidx[i])] - self.rsgFltBU_MeanError, dx=1/100)                                                                       
            
            self.theta1 = self.theta1 - self.theta1[0]
            self.theta2 = self.theta2 - self.theta2[0]
            self.theta1R = self.theta1R - self.theta1R[0]
            self.theta2R = self.theta2R - self.theta2R[0]     
            
            startStepX = l1*np.sin(self.theta1R[int((len(self.theta1)-1)/2)]*np.pi/180 - self.theta1R[0]*np.pi/180) + l2*np.sin(self.theta2R[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2R[0]*np.pi/180) + l1*np.sin(self.theta1[0]*np.pi/180) + l2*np.sin(self.theta2[0]*np.pi/180 - self.theta1[0]*np.pi/180)
            startStepY = l1*np.cos(self.theta1R[int((len(self.theta1)-1)/2)]*np.pi/180 - self.theta1R[0]*np.pi/180) + l2*np.cos(self.theta2R[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2R[0]*np.pi/180) - (l1*np.cos(self.theta1[0]*np.pi/180) + l2*np.cos(self.theta2[0]*np.pi/180 - self.theta1[0]*np.pi/180))
            
            endStepX = l1*np.sin(self.theta1R[int((len(self.theta1)-1)/2)]*np.pi/180 - self.theta1R[len(self.theta1)-1]*np.pi/180) + l2*np.sin(self.theta2R[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2R[len(self.theta1)-1]*np.pi/180) + l1*np.sin(self.theta1[len(self.theta1)-1]*np.pi/180) + l2*np.sin(self.theta2[len(self.theta1)-1]*np.pi/180 - self.theta1[len(self.theta1)-1]*np.pi/180)
            endStepY = l1*np.cos(self.theta1R[int((len(self.theta1)-1)/2)]*np.pi/180 - self.theta1R[len(self.theta1)-1]*np.pi/180) + l2*np.cos(self.theta2R[int((len(self.theta1)-1)/2)]*np.pi/180-self.theta2R[len(self.theta1)-1]*np.pi/180) - (l1*np.cos(self.theta1[len(self.theta1)-1]*np.pi/180) + l2*np.cos(self.theta2[len(self.theta1)-1]*np.pi/180 - self.theta1[len(self.theta1)-1]*np.pi/180))

            sumLeftLength += abs(startStepX) + abs(endStepX)                        
            # print('x value: {}'.format(abs(startStepX) + abs(endStepX)))
            # print('y value: {}'.format(abs(abs(endStepY) - abs(startStepY))))        
                      
        self.results.totRightDistance = sumRightLength
        self.results.totLeftDistance = sumLeftLength
        
        if np.mean([sumRightLength,sumLeftLength]) > 0:
            self.results.totDistance = np.mean([sumRightLength,sumLeftLength])
        else:
            self.results.totDistance = 0
        print('Total distance - right leg: {}'.format(self.results.totRightDistance))
        print('Total distance - left leg: {}'.format(self.results.totLeftDistance))            
        print('Total distance: {}'.format(self.results.totDistance))
#################################################################











                         # YO #












########################## MAIN LÖÖP #############################
##################################################################
if __name__ == "__main__":
    # import sys
    # sys.path.insert(1,'/Users/bjorgvinbjorgvinsson/Documents/Myoswiss/Myosense/Code/data-processing-gamma/proto_parsers')
    # import deserializer
    # from deserializer import data_log
    # data_log = deserializer.data_log
    
    all_data = dashboard_metrics(read_metric, 1)#.resultDF)  
    # ## Run the additional processing of the metrics ##
    all_data.processWalking() # Get heel strikes, toe offs
    all_data.processMotorData() # Get RMS currents during the walking
    all_data.processCadence() # Get cadence-related data 
    all_data.processSTS() # Get Sitting-transfer related data
    
    all_data.processStairs() # Get stairs-related data
    # all_data.processStairs([read_metric.CalAngMaxR/1000, read_metric.CalAngMaxL/1000]) # Get stairs-related data
    # all_data.myoScore()
    all_data.processIntervals() # Get the best performing interval data
    all_data.processDistance()
    metrics_calculated = all_data.results
    print("All DONE!")    