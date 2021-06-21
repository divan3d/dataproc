# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:15:18 2020

performs calibration of load cells in 3 different ways.

1. weight based: instructs user to put weights on the load cell to calibrate it statically
2. reference based: uses an already calibrated load cell
3. from file: loads a lookup table from previous calibration, must be in the same folder

@author: Marc
"""

import serial
import time
import numpy as np
import sys
from scipy import interpolate
import datetime
import pickle
from scipy.signal import medfilt

#  Choose COM channel
COM = 'COM3'

#initialize empty lookup tables
lookupLC1 = np.zeros([1024,2])
lookupLC2 = np.zeros([1024,2])

#set the weights for the static calibration
weights = np.array([0.,2.5,5.,10.,15.,20.,25.,30.]) #in kg

#tranform weigths to Newtons
Newts = weights * 9.81

#input type of calibration of force cell 1
calibtype1 = int(input("Calibration Force Cell 1: None (0), Weight based (1), Reference based(2), from file (3)"))
if calibtype1 == 0:
    print('Force Cell 1 is not calibrated')
if calibtype1 == 1:
    print('Force Cell 1 is calibrated based on static weights')
if calibtype1 == 2:
    print('Force Cell 1 is calibrated based on redundant force cell measurements')
if calibtype1 == 3:
    print('Force Cell 1 is not calibrated. Data is loaded from an existing file')   

time.sleep(1)

#input type of calibration of force cell 2, check feasibility
calibtype2 = int(input("Calibration Force Cell 2: None (0), Weight based (1), Reference based(2), from file (3)"))
if calibtype2 == 0:
    if calibtype1 == 2:
        print('Invalid combination chosen, please restart the script')
        sys.exit()
    elif calibtype1 != 2:
        print('Force Cell 2 is not calibrated')
if calibtype2 == 1:
    print('Force Cell 2 is calibrated based on static weights')
if calibtype2 == 2:
    if calibtype1 == 1:
        print('Force Cell 2 is calibrated based on redundant force cell measurements')
    elif calibtype1 ==0 or calibtype1 ==2:
        print('Invalid combination chosen, please restart the script')
        sys.exit()
if calibtype1 == 3:
    print('Force Cell 2 is not calibrated. Data is loaded from an existing file') 

time.sleep(1)

#load lookup tables if the calibration should be loaded
if calibtype1 == 3:
    with open('lookup_FC_1','rb') as fileopen:
        lookupLC1=pickle.load(fileopen)
        
if calibtype2 == 3:
    with open('lookup_FC_2','rb') as fileopen:
        lookupLC2=pickle.load(fileopen)
        
#weight based calibration for force sensor 1
if calibtype1 == 1:
    print('Static calibration of force cell 1 started')
    
    #initialize empty array to store the discrete load cell data
    bytes1 = np.zeros([len(weights)])  
    
    #iterate through all weights
    for i in range(0,len(weights)):
        
        #give instructions
        print('Attach weight:', weights[i], 'kg')    
        proceed = input("Are you ready? (y/n)")
        
        #initialize running variables
        counter = 0
        countertot = 0
        bytes1sum = 0
                
        if proceed == "y":
            #open serial connection
            try:
                ser = serial.Serial(COM)
                ser.flushInput()
            except Exception as eOpen:
                print("Could not open the serial port, try again")
                ser = serial.Serial(COM)
                ser.close()
                time.sleep(0.1)
                ser.flushInput()    
            
            #record force cell data for the first 500 measurements
            while 1:  
                
                #increase counter to determine when to stop calibration
                countertot += 1
                
                                
                try:
                    #  Read serial data
                    ser_bytes = ser.readline()
                    ser_bytes = ser_bytes.decode()
                    ser_bytes = ser_bytes.rstrip()
                    decoded_bytes_1 = int(ser_bytes[0:3])
                    print(decoded_bytes_1)                    
                    bytes1sum += decoded_bytes_1
                        
                    #counter to determine when to output a message
                    counter += 1
                    if counter == 20:
                        counter = 0
                        #  Print to terminal
                        print(decoded_bytes_1)
            
                    time.sleep(0.01)
                
                except Exception as excpt:
                    print("Exception : {}".format(excpt))
                    time.sleep(0.03)
                    ser.close()
                    ser = serial.Serial(COM)
                    continue
                
                #stop if counter reaches 500 
                if countertot == 500:
                    break
            
            #close serial connection
            ser.close()
            #average input for defined weight
            bytes1[i] = bytes1sum/500
            
    #build lookup table
    lookupLC1[:,0] = np.arange(0,1024,1)
    
    #build vector which contains the measured load cell counts between the end points 0 and 1023
    bytesinterp = np.zeros(len(bytes1)+2)
    bytesinterp[1:-1] = bytes1[:]
    bytesinterp[-1] = lookupLC1[-1,0]
    
    #build vector with the assigned forces. The first and the last entries are extrapolations of the forces based on the bytes vector
    forceinterp = np.zeros(len(Newts)+2)
    forceinterp[1:-1] = Newts[:]
    forceinterp[0] = Newts[0] - (Newts[-1]-Newts[0])/(bytes1[-1]-bytes1[0])*(bytesinterp[1]-bytesinterp[0])
    forceinterp[-1] = Newts[len(Newts)-1] + (Newts[-1]-Newts[0])/(bytes1[-1]-bytes1[0])*(bytesinterp[-1]-bytesinterp[-2])

    #interpolate to build lookup table
    intp = interpolate.interp1d(bytesinterp,forceinterp,fill_value="extrapolate")    
    lookupLC1[:,1] = intp(lookupLC1[:,0])
        
        
            
#weight based calibration for force sensor 2
if calibtype2 == 1:
    
    print('Static calibration of force cell 2 started') 
    
    #initialize empty array to store the discrete load cell data
    bytes2 = np.zeros([len(weights)])
    
    #iterate through all weights
    for i in range(0,len(weights)):
        
        #give instructions
        print('Attach weight:', weights[i], 'kg')    
        proceed = input("Are you ready? (y/n)")
        
        #initialize running variables
        counter = 0
        countertot = 0
        bytes2sum = 0
        
        if proceed == "y":
            #open serial connection
            try:
                ser = serial.Serial(COM)
                ser.flushInput()
            except Exception as eOpen:
                print("Could not open the serial port, try again")
                ser = serial.Serial(COM)
                ser.close()
                time.sleep(0.1)
                ser.flushInput()    

            #record force cell data for the first 500 measurements
            while 1:
                
                #increase counter to determine when to stop calibration
                countertot += 1
                if countertot == 500:
                    break                
                
                try:
                    #  Read serial data
                    ser_bytes = ser.readline()
                    ser_bytes = ser_bytes.decode()
                    ser_bytes = ser_bytes.rstrip()
                    decoded_bytes_2 = int(ser_bytes[3:6])
                    
                    bytes2sum += decoded_bytes_2
    
                    #counter to determine when to output a message
                    counter += 1
                    if counter == 20:
                        counter = 0
                        #  Print to terminal
                        print(decoded_bytes_1)
                    
                    time.sleep(0.01)
                    
                
                except Exception as excpt:
                    print("Exception : {}".format(excpt))
                    time.sleep(0.03)
                    ser.close()
                    ser = serial.Serial(COM)
                    continue
                
            #close serial connection
            ser.close()
            #average input for defined weight
            bytes2[i] = bytes2sum/500
    
    #build lookup table
    lookupLC2[:,0] = np.arange(0,1024,1)
    
    #build vector which contains the measured load cell counts between the end points 0 and 1023
    bytesinterp = np.zeros(len(bytes2)+2)
    bytesinterp[1:-1] = bytes2[:]
    bytesinterp[-1] = lookupLC2[-1,0]
    
    #build vector with the assigned forces. The first and the last entries are extrapolations of the forces based on the bytes vector
    forceinterp = np.zeros(len(Newts)+2)
    forceinterp[1:-1] = Newts[:]
    forceinterp[0] = Newts[0] - (Newts[-1]-Newts[0])/(bytes2[-1]-bytes2[0])*(bytesinterp[1]-bytesinterp[0])
    forceinterp[-1] = Newts[len(Newts)-1] + (Newts[-1]-Newts[0])/(bytes2[-1]-bytes2[0])*(bytesinterp[-1]-bytesinterp[-2])

    #interpolate to build lookup table
    intp = interpolate.interp1d(bytesinterp,forceinterp,fill_value="extrapolate")    
    lookupLC2[:,1] = intp(lookupLC2[:,0])
 
#check if one sensor is calibrated by using the other sensor as reference       
if calibtype1 == 2 or calibtype2 == 2:
    bytes1 = []
    bytes2 = []
    if calibtype1 == 2:
        print('Reference calibration of force cell 1 started')
    if calibtype2 == 2:
        print('Reference calibration of force cell 2 started')
    
    #input measurment time in seconds
    t = int(input("Type the number of seconds which should be used for calibration"))
    proceed = input("press Enter when you are ready to start the measurement")
    
    if proceed == "":
        #open serial connection
        try:
            ser = serial.Serial(COM)
            ser.flushInput()
        except Exception as eOpen:
            print("Could not open the serial port, try again")
            ser = serial.Serial(COM)
            ser.close()
            time.sleep(0.1)
            ser.flushInput()    
            
        #start the time
        start = datetime.datetime.now()
        stop = 0
        counter = 0
        
        #check if time is over
        while stop < t:
            
            try:
                #  Read serial data
                ser_bytes = ser.readline()
                ser_bytes = ser_bytes.decode()
                ser_bytes = ser_bytes.rstrip()
                decoded_bytes_1 = int(ser_bytes[0:3])
                decoded_bytes_2 = int(ser_bytes[3:6])
                
                bytes1.append(decoded_bytes_1)
                bytes2.append(decoded_bytes_2)
                
                counter += 1
                if counter == 20:
                    counter = 0
                    #  Print to terminal
                    print(decoded_bytes_1,decoded_bytes_2)
                
                time.sleep(0.01)
            
            except Exception as excpt:
                print("Exception : {}".format(excpt))
                time.sleep(0.03)
                ser.close()
                ser = serial.Serial(COM)
                continue
            
            #update time
            stop = datetime.datetime.now() - start
            stop = stop.seconds
            
        #transform list to numpy array
        bytes1 = np.asarray(bytes1)
        bytes2 = np.asarray(bytes2)
        
        #check which sensor should be calibrated
        if calibtype1 == 2:
            
            #fill a vector with data of the redundant sensor
            forces2 = np.zeros([len(bytes2)])            
            for i in range(0,len(forces2)):
                forces2[i]=lookupLC2[bytes2,1]
            
            #fill first column of lookup table
            lookupLC1[:,0] = np.arange(0,1024,1)
            
            temp = []
            #look for same entries in the measurements and assign an average force to this value based on the data of the redundant sensor
            for i in range(0,1024):
                indices = np.where(bytes1==i)[0]
                if len(indices)!=0:
                    temp.append([i,np.sum(forces2[indices])/len(indices)])
            temp=np.asarray(temp)
            
            #interpolate the data to get the full lookup table
            bytesintp = np.zeros(len(temp)+2)
            bytesintp[1:-1] = temp[:,0]
            bytesintp[-1] = 1023
            
            forceintp = np.zeros(len(temp)+2)
            forceintp[1:-1] = temp[:,1]
            forceintp[0] = temp[0,1] - (temp[-1,1]-temp[0,1])/(temp[-1,0]-temp[0,0])*temp[0,0]
            forceintp[-1] = temp[-1,1] + (temp[-1,1]-temp[0,1])/(temp[-1,0]-temp[0,0])*(bytesintp[-1]-temp[-1,0])
                        
            intp = interpolate.interp1d(bytesintp,forceintp,fill_value="extrapolate")
            
            lookupLC1[:,1] = intp(lookupLC1[:,0])
            
            #filter the interpolated data
            lookupLC1[:,1] = medfilt(lookupLC1[:,1],21)
            
        #check which sensor should be calibrated
        if calibtype2 == 2:
            
            #fill a vector with data of the redundant sensor
            forces1 = np.zeros([len(bytes1)])
            
            for i in range(0,len(forces1)):
                forces1[i]=lookupLC1[bytes1[i],1]
            
            #fill first column of lookup table    
            lookupLC2[:,0] = np.arange(0,1024,1)
            
            temp = []
            #look for same entries in the measurements and assign an average force to this value based on the data of the redundant sensor
            for i in range(0,1024):
                indices = np.where(bytes2==i)[0]
                if len(indices)!=0:
                    temp.append([i,np.sum(forces1[indices])/len(indices)])
            temp=np.asarray(temp)
            
            #interpolate the data to get the full lookup table
            bytesintp = np.zeros(len(temp)+2)
            bytesintp[1:-1] = temp[:,0]
            bytesintp[-1] = 1023
            
            forceintp = np.zeros(len(temp)+2)
            forceintp[1:-1] = temp[:,1]
            forceintp[0] = temp[0,1] - (temp[-1,1]-temp[0,1])/(temp[-1,0]-temp[0,0])*temp[0,0]
            forceintp[-1] = temp[-1,1] + (temp[-1,1]-temp[0,1])/(temp[-1,0]-temp[0,0])*(bytesintp[-1]-temp[-1,0])
                        
            intp = interpolate.interp1d(bytesintp,forceintp,fill_value="extrapolate")
            
            lookupLC2[:,1] = intp(lookupLC2[:,0])
            
            #filter the interpolated data
            lookupLC2[:,1] = medfilt(lookupLC2[:,1],21)

    #close serial connection
    ser.close()

#save files
with open('lookup_FC_1','wb') as fp:
    pickle.dump(lookupLC1,fp)

with open('lookup_FC_2','wb') as fp:
    pickle.dump(lookupLC2,fp)


            
            
        
                
    
        
        
