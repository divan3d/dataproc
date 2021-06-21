# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:32:48 2018

@author: Gleb Koginov
"""
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pandas as pd
import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt
from os import path, getcwd
import c3dMS as c3d
from re import search, findall
import pickle
import sys
from glob import glob


from datetime import datetime

   

class viconProcessor():
    
    def __init__(self, viconFile):
        self.points = []  # 13 markers, 5 values (x,y,z, residual value(estimate of the error for this particular point), cameras value(number of cameras that observed the point))
        self.vgrf = [] # vertical ground reaction force
        self.Fz1 = []
        self.Fz2 = []
        #self.Fz1_INTERP = []
        #self.Fz2_INTERP = []
        self.MARKER_DATA = {}
        
        # Main angles #
        #self.C_TRUNK_ANGLE = []
        #self.L_SHANK_ANGLE = []
        #self.R_SHANK_ANGLE = []
        #self.L_THIGH_ANGLE = []
        #self.R_THIGH_ANGLE = []
        
        # Read in the data #
        self.viconFile = viconFile
        self.readFile()
        
        # Extract the marker data #
        self.marker_coordinates()
        
        # Get the number of samples in the self.points array and the self.vgrf array (difference comes from the different fs)
        nSamplesPTS = self.points_array.shape[0]
        #nSamplesVGRF =  self.vgrf.shape[1]
        self.TIME_VECTOR = np.arange(0, nSamplesPTS * (1/100), (1/100)) # [s], new common time vector
        
        # Downsample the vgrf vectors #
        #self.vgrfTimeAxis = np.arange(0, nSamplesVGRF * (1/1000), (1/1000)) # [s]
        #self.Fz1_INTERP = self.interpolateData(self.vgrf[0,:], self.vgrfTimeAxis)
        #self.Fz2_INTERP = self.interpolateData(self.vgrf[1,:], self.vgrfTimeAxis)
        
        # Add the angle data #
#        self.C_TRUNK_ANGLE = self.calcAngle(self.MARKER_DATA['C_ACRO'], self.MARKER_DATA['C_TRC']) # Trunk angle
#        self.L_THIGH_ANGLE = self.calcAngle(self.MARKER_DATA['L_TRC'],  self.MARKER_DATA['L_KNEE']) # Left thigh angle
#        self.R_THIGH_ANGLE = self.calcAngle(self.MARKER_DATA['R_TRC'],  self.MARKER_DATA['R_KNEE']) # Right thight angle
#        self.L_SHANK_ANGLE = self.calcAngle(self.MARKER_DATA['L_KNEE'], self.MARKER_DATA['L_ANK']) # Left shank angle
#        self.R_SHANK_ANGLE = self.calcAngle(self.MARKER_DATA['R_KNEE'], self.MARKER_DATA['R_ANK']) # Right shank angle

    def readFile(self):    # Read in the .c3d file, return ground reaction forces, point cloud and marker self.labels #
        reader = c3d.Reader(open(self.viconFile,'rb'))
        self.labels = reader.point_labels # Extract the marker self.labels (TODO: leave only the correct markers)
        self.labels = list(map(str.strip, self.labels))
        #self.labels = [str(x) for x in self.labels] # For compatibility with Python 2
        
        for i, point, analog in reader.read_frames(): # Enumerate through the frames to get the data
            self.points.append(point)
            self.vgrf.append(analog) # Usually sampled at higher frequency than Vicon - make sure the rotation matrix is correct in the library, use myoswiss library only
        self.points_array = np.array(self.points)
        
        # Extract only the vertical ground reaction forces (discard the other axes and moments)
        self.vgrf = np.hstack(self.vgrf)
        #self.vgrf = np.vstack((self.vgrf[2,:],  self.vgrf[8,:]))
        
        #self.Fz1 = self.vgrf[0,:]
        #self.Fz2 = self.vgrf[1,:]
        
        
    def interpolateData(self, rawDatas, rawTime):
    # Interpolate the data to be between the common data points #
        
        dataInterp = np.empty((self.TIME_VECTOR.shape[0], 2), dtype = float)
        dataInterp[:,0] = self.TIME_VECTOR
        
        interpolator = interp.interp1d(rawTime, rawDatas) 
        dataInterp[:,1] = interpolator(self.TIME_VECTOR)
            
        return dataInterp # out
    
    def marker_coordinates(self):
        
        # Saving markers coordinates per element for joint angle calculation
        # self.MARKER_DATA['L_ACRO'] = self.points_array[:,self.labels.index('L_ACRO'),0:3] # Left Shoulder
        # self.MARKER_DATA['R_ACRO'] = self.points_array[:,self.labels.index('R_ACRO'),0:3] # Right Shoulder
        
        # self.MARKER_DATA['L_ASIS'] = self.points_array[:,self.labels.index('L_ASIS'),0:3] # Left Hip Bone
        # self.MARKER_DATA['R_ASIS'] = self.points_array[:,self.labels.index('R_ASIS'),0:3] # Right Hip Bone
        
        # self.MARKER_DATA['L_TRC'] = self.points_array[:,self.labels.index('L_TROC'),0:3] # Left Trochanter
        # self.MARKER_DATA['R_TRC'] = self.points_array[:,self.labels.index('R_TROC'),0:3] # Right Trochanter
        
        # self.MARKER_DATA['L_KNEE'] = self.points_array[:,self.labels.index('L_KNEE'),0:3] # Left Knee
        # self.MARKER_DATA['R_KNEE'] = self.points_array[:,self.labels.index('R_KNEE'),0:3] # Right Knee
        
        # self.MARKER_DATA['R_ANK'] = self.points_array[:,self.labels.index('R_ANKL'),0:3] # Right Ankle
        # self.MARKER_DATA['L_ANK'] = self.points_array[:,self.labels.index('L_ANKL'),0:3] # Left Ankle
        
        # self.MARKER_DATA['L_MT5'] = self.points_array[:,self.labels.index('L_MT5'),0:3] # Left MT5
        # self.MARKER_DATA['R_MT5'] = self.points_array[:,self.labels.index('R_MT5'),0:3] # Right MT5
        
        # self.MARKER_DATA['L_MT1'] = self.points_array[:,self.labels.index('L_MT1'),0:3] # Left MT1
        # self.MARKER_DATA['R_MT1'] = self.points_array[:,self.labels.index('R_MT1'),0:3] # Right MT1
        
        # self.MARKER_DATA['L_THIGH_IMU'] = self.points_array[:,self.labels.index('L_THIGH_IMU'),0:3] # Left Upper IMU
        # self.MARKER_DATA['R_THIGH_IMU'] = self.points_array[:,self.labels.index('R_THIGH_IMU'),0:3] # Right Upper IMU
        
        # self.MARKER_DATA['L_SHIN'] = self.points_array[:,self.labels.index('L_SHIN'),0:3] # Left Shin
        # self.MARKER_DATA['R_SHIN'] = self.points_array[:,self.labels.index('R_SHIN'),0:3] # Right Shin
        
        # self.MARKER_DATA['L_SHANK_IMU'] = self.points_array[:,self.labels.index('L_SHANK_IMU'),0:3] # Left Lower IMU
        # self.MARKER_DATA['R_SHANK_IMU'] = self.points_array[:,self.labels.index('R_SHANK_IMU'),0:3] # Right Lower IMU
                
        # self.MARKER_DATA['C_VERT'] = self.points_array[:,self.labels.index('C_VERT'),0:3] # Neck Vertebrea
        
        # self.MARKER_DATA['C_TRC'] = (self.points_array[:,self.labels.index('R_TROC'),0:3] + self.points_array[:,self.labels.index('L_TROC'),0:3])/2  # Centre of Trochanter
        # self.MARKER_DATA['C_ACRO'] = (self.points_array[:,self.labels.index('R_ACRO'),0:3] + self.points_array[:,self.labels.index('L_ACRO'),0:3])/2  # Centre of Shoulder
        self.MARKER_DATA = {}
        for i in range(0, len(self.labels)):
            self.MARKER_DATA[self.labels[i]] = self.points_array[:,self.labels.index(self.labels[i]),0:3]
            
            
            
    def calcAngle(self, marker1, marker2):
        # Calculates the angle between a two-point vector and the horizontal line #
        # The horizontal line is always computed from the lower marker #
        # Always use the "marker2" input as the lower of the two marker coordinates #
        # Here walking is in the (-y) direction and the z-direction is from ground-up
        
        rad2deg = 180/np.pi # Conversion from rad to degrees
        
        # Construct a horizontal vector #
        horizntY = marker2[:, 1] - 100 # Horizontal - any length will do, shouldn't matter 
        horizntZ = marker2[:, 2] # Vertical position is the same
        
        vectorMan = np.vstack((marker1[:, 1] - marker2[:,1], marker1[:, 2] - marker2[:, 2])).T # Limb vector 
        vectorHor = np.vstack((horizntY - marker2[:, 1], horizntZ - marker2[:, 2])).T # Horizontal vector 
        
        dotProd = (vectorMan*vectorHor).sum(1) # Vector dot product 
        
        lenVectMan = np.sqrt((vectorMan[:, 0] ** 2) + (vectorMan[:, 1] ** 2)) # Limb vector length
        lenVectHor = np.sqrt((vectorHor[:, 0] ** 2) + (vectorHor[:, 1] ** 2)) # Horizontal vector length 
        
        vectAngle = np.arccos(dotProd / (lenVectMan * lenVectHor)) # Angle between the two vectors
        
        return vectAngle * rad2deg
    

    def assemblePkl(self):
        # Assemble the required data for saving #
        postProcOut = {}
        #postProcOut['C_TRUNK_ANGLE'] = self.C_TRUNK_ANGLE
        #postProcOut['Fz1'] = self.Fz1
        #postProcOut['Fz2'] = self.Fz2
        #postProcOut['Fz1_INTERP'] = self.Fz1_INTERP
        #postProcOut['Fz2_INTERP'] = self.Fz2_INTERP
        #postProcOut['L_SHANK_ANGLE'] = self.L_SHANK_ANGLE
        #postProcOut['L_THIGH_ANGLE'] = self.L_THIGH_ANGLE
        #postProcOut['R_SHANK_ANGLE'] = self.R_SHANK_ANGLE
        #postProcOut['R_THIGH_ANGLE'] = self.R_THIGH_ANGLE
        postProcOut['time'] = self.TIME_VECTOR
        postProcOut['labels'] = self.labels
        postProcOut['points'] = self.points
        postProcOut['points_array'] = self.points_array
        postProcOut['vgrf'] = self.vgrf
        
        return postProcOut
def extraction(foldername):
    ############################################### Process the GRAIL data ###############################################

    # Source folder & files #
    sessionID = [foldername]
    
    # Run the processing #
    tduData = {}
    grailData = {}    
    for i in range(0,len(sessionID)):
        print("\nProcessing for SessionID: {}".format(sessionID[i]))
        
        sessFP = path.join(dname, sessionID[i])
        rawGRAILData = glob(sessFP + '\\*.c3d')
        
        print("\nGRAIL FILES...")
        for k in range(0, len(rawGRAILData)):
        
            f = rawGRAILData[k].split('\\')[-1]        
            fileRef = f[0:-4] # reference name used         
            f2Wr = path.join(sessFP, fileRef+'.pkl')
            
            print("\nProcessing GRAIL file: {}".format(f))
            procObj = viconProcessor(rawGRAILData[k])
            grailData[sessionID[i] + "_" + fileRef] = procObj # Pre-define the trial a specific dict field 
        
            print("Processing suscessful. Saving the processed data as {}.pkl".format(fileRef))
            with open(f2Wr, 'wb') as outfile:
                pickle.dump(procObj.assemblePkl(), outfile, pickle.HIGHEST_PROTOCOL)
    
    return procObj

if __name__ == "__main__":
    
    filepath = r"C:\Users\wirth\polybox\MSc\Master Thesis\03_Relevant Files\03_Software\MoCap_Logs\S01_19112020_Corrected"    
    procObj = extraction(filepath)
    