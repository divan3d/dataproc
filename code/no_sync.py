# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:43:28 2021

@author: ksdiv
for SE - missing sync for FL3, FL5_1
"""


import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy import interpolate as itp
import scipy.signal
import statistics
from scipy import stats

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#MS
filee5 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SE\MS_log_SE_FL3.pkl"
data1 = op_pickle(filee5)

#mocap 
filmce5 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\SE\INTERMEDIATE\SE_FUN_FL3.pkl"
datamc = op_pickle(filmce5)

#%%
force_level = data1.sensorS_StateRightForce

# proximal 
force  = data1["extSensorS_Analog0"]
# MS 100 Hz - qd passe de 0 a 1 - début du recording, qd repasse a 0 c'est juste qu'on a viré le cable 
sync = data1["extSensorS_SyncIn"]
# time a changer pr que mets ensemble et garde jusqu'a je ne sais quelle valeur
time_s = data1["tv_sec"] 
time_n = data1["tv_nsec"]
# celui la qu'on garde 
time = time_s + time_n*1e-9
# mode faut jouer entre les 2 valeurs ici pr savoir qd c'est tranparent, isometric ou concentric
mode = data1["sensorS_StateSysMode"]
sysinf = data1["sensorS_StateSysInf"]
sysinf = sysinf - 8
Mode = mode * sysinf
Mode = Mode.replace(0,3)
# courant - regarde que jambe droite 
# qd negatif - juste que ça tourne ds direction opposée 
current_sent = data1["CurrentRightMotor"]
current_received = data1["ActualCurrentRightMotor"]
# Rleg, L leg 
R_Leg = data1["sensorS_StateRightStanceDetection"]
L_Leg = data1["sensorS_StateLeftStanceDetection"]
# interpreted angles
AlphaShank = data1["IMURightShankKalmanAngle"]
AlphaThigh = data1["IMURightThighKalmanAngle"]
AlphaTrunk = data1["IMUTrunkKalmanAngle"]
# calibration - s'en fou toute facon donne que 0 
RightCalibration = data1["commandS_rxCalibration"]
LeftCalibration = data1["commandS_lxCalibration"]
# hall sensor - utilise ? 
HallSensor = data1["HallRightMotor"]


# plt.figure()
# plt.plot(sync*10)
# plt.plot(R_Leg)
# plt.plot(L_Leg)
# plt.plot(AlphaShank/10000 - 8)
# plt.title("E3")

# #%%
# plt.figure()
# plt.plot(current_received)
# plt.plot(AlphaShank)
# plt.plot(current_sent)
# plt.title("current")

# plt.figure()
# plt.plot(AlphaShank)
# plt.plot(AlphaThigh)
# plt.title("angles imu")
# # plt.plot(AlphaTrunk)

# #%%

# plt.figure()
# plt.plot(datamc["no_mc_shank_angle"])
# #%%
# plt.figure()
# plt.plot(R_Leg)
# plt.plot(L_Leg)

# plt.figure()
# plt.plot(datamc["vgrf"])

#%%

new_sync = sync
new_sync[3232] = 1 
CurrentSent = current_sent
CurrentRead = current_received
R_leg = R_Leg
L_leg = L_Leg
timeMS = time

t = datamc["t"]

indexMS = new_sync.argmax()
        
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
# Force = Force[indexMS:]
# ForceLevel = ForceLevel[indexMS:]
# GyroCThigh = GyroCThigh[indexMS:]
# GyroCShank = GyroCShank[indexMS:]
# AccelAThigh = AccelAThigh[indexMS:]
# AccelAShank = AccelAShank[indexMS:]
            
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
# interForce = itp.interp1d(timeMS, Force)
# interFL = itp.interp1d(timeMS, ForceLevel)
# interGCT = itp.interp1d(timeMS,GyroCThigh)
# interGCS = itp.interp1d(timeMS,GyroCShank)
# interACT = itp.interp1d(timeMS,AccelAThigh)
# interACS = itp.interp1d(timeMS,AccelAShank)

# c'était necessaire pr SD_FL5, parce que données MS se coupent soudainement
t = t[t<timeMS.iloc[-1]]
    

AlphaShank = interAShank(t)
AlphaThigh = interAThigh(t)
AlphaTrunk = interATrunk(t)
# EncCount   = interEC(t)
CurrentRead = interCR(t)
CurrentSent = interCS(t)  
R_Leg = interR_leg(t)
L_Leg = interL_leg(t)
Mode = interMode(t)
# Force = interForce(t)
# ForceLevel = interFL(t)
# GyroCShank = interGCS(t)
# GyroCThigh = interGCT(t)
# AccelAShank = interACS(t)
# AccelAThigh = interACT(t)

# AlphaShank = AlphaShank[:40000]
# AlphaThigh = AlphaThigh[:40000]
# AlphaTrunk = AlphaTrunk[:40000]
# # EncCount   = interEC(t)
# CurrentRead = CurrentRead[:40000]
# CurrentSent = CurrentSent[:40000]
# R_Leg = R_Leg[:40000]
# L_Leg = L_Leg[:40000]
# Mode = Mode[:40000]

# datamc = datamc[:40000]

#%% check if everything makes sense - especially myosuit vs mocap data 

datamc = datamc[:len(t)]

def plot_Fig1(dict_in):
    plt.figure()
    plt.plot(dict_in["t"], dict_in["vgrf"] -550, label = "vgrf")
    plt.plot(dict_in["t"], dict_in["no_mc_shank_angle"], label = "shank mocap")
    plt.plot(dict_in["t"], dict_in["no_mc_kmal_angle"], label = "kmal mocap")
    plt.plot(dict_in["t"], AlphaShank/1000 -80, label = "shank myosuit")
    plt.plot(dict_in["t"], R_Leg *100, label = "stance R leg")
    plt.plot(dict_in["t"], L_Leg *100, label = "stance L leg")
    plt.legend()
    plt.title("FIG 1: vgrf, shank angles")
    return 

# def plot_Fig1(dict_in):
#     plt.figure()
#     plt.plot(t, dict_in["vgrf"] -550, label = "vgrf")
#     plt.plot(t, dict_in["no_mc_shank_angle"], label = "shank mocap")
#     plt.plot(t, dict_in["no_mc_kmal_angle"], label = "kmal mocap")
#     plt.plot(t, AlphaShank/1000 -80, label = "shank myosuit")
#     plt.plot(t, R_Leg *100, label = "stance R leg")
#     plt.plot(t, L_Leg *100, label = "stance L leg")
#     plt.legend()
#     plt.title("FIG 1: vgrf, shank angles")
#     return 

plot_Fig1(datamc)