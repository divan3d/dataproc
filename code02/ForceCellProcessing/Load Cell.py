import serial
import time
import csv
import numpy as np
import pickle

# Choose COM channel
COM = 'COM3'

# Declare file name where data will be saved (CSV)
file_name = "S06_23112020_T02.csv"

#open load cell lookup tables
with open('lookup_FC_1','rb') as fileopen:
    lookupLC1=pickle.load(fileopen)
with open('lookup_FC_2','rb') as fileopen:
    lookupLC2=pickle.load(fileopen)
   
#open serial port     
try:
    ser = serial.Serial(COM)
    ser.flushInput()
except Exception as eOpen:
    print("Could not open the serial port, try again")
    ser = serial.Serial(COM)
    ser.close()
    time.sleep(0.1)
    ser.flushInput()
    

# get time now. Used as a reference for calculating time per iteration
time_start = time.time()

#  while-loop to constantly read incoming data. Must abort manually.
counter = 0
with open(file_name,"a") as file:
    while True:
    
        try:
            #  Read serial data
            counter+=1
            ser_bytes = ser.readline()
            ser_bytes = ser_bytes.decode()
            ser_bytes = ser_bytes.rstrip()
            decoded_bytes_1 = int(ser_bytes[0:3])
            decoded_bytes_2 = int(ser_bytes[3:6])
            sync = int(ser_bytes[6])
            
            #read force values out of the lookup tables            
            Force1 = lookupLC1[decoded_bytes_1,1]
            Force2 = lookupLC2[decoded_bytes_2,1]
            
            #print to terminal every 20 measurements
            counter += 1            
            if counter == 20:
                counter = 0
                #  Print to terminal
                print('Force 1: {:+0.2f}, Force 2: {:+0.2f}, Sync: {}'.format(Force1, Force2, sync))

            #  Open
            writer = csv.writer(file, delimiter=",")
            writer.writerow([time.time()-time_start, Force1, Force2, sync])
        
        except Exception as excpt:
            print("Exception : {}".format(excpt))
            time.sleep(0.03)
            ser.close()
            ser = serial.Serial(COM)
            continue

