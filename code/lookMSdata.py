# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:00:58 2021

@author: ksdiv

regarde données ds nv logs MS
"""


import pickle
import numpy as np
import math as m
import matplotlib.pyplot as plt
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
        
        
# file1 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SA\MS_log_SA_FL5_2.pkl"
# data1 = op_pickle(file1)
        
# filec5 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SC\MS_log_SC_FL5.pkl"
# data1 = op_pickle(filec5)

# filec1 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SC\MS_log_SC_FL1.pkl"
# data1 = op_pickle(filec1)
        
# filec3 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SC\MS_log_SC_FL3.pkl"
# data1 = op_pickle(filec3)

# filee3 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SE\MS_log_SE_FL3.pkl"
# data1 = op_pickle(filee3)

filee1 = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SE\MS_log_SE_FL1.pkl"
data1 = op_pickle(filee1)


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

#%%

plt.figure()
plt.plot(sync*10)
plt.plot(R_Leg)
plt.plot(L_Leg)
# plt.plot(AlphaShank/10000 - 8)
plt.title("C5")

#%%
plt.figure()
plt.plot(current_received)
plt.plot(current_sent)
plt.title("current")

plt.figure()
plt.plot(AlphaShank)
plt.plot(AlphaThigh)
plt.title("angles imu")
# plt.plot(AlphaTrunk)

plt.figure()
plt.plot(sync)
plt.title("sync")

plt.figure()
plt.plot(force)
plt.title("force")


plt.figure()
plt.plot(Mode)
plt.title("Mode")

plt.figure()
plt.plot(R_Leg)
plt.plot(L_Leg)
plt.title(" stance detection")

plt.figure()
plt.plot(time)
plt.title("time")

plt.figure()
plt.plot(force_level)
plt.title("force level")

#%%
plt.figure()
plt.plot(AlphaShank/10000)
plt.plot(AlphaThigh/10000)
plt.title("angles imu")
# plt.plot(AlphaTrunk)
#%%
plt.figure()
plt.plot(current_received/1000)
plt.plot(current_sent/1000)
plt.plot(sync)
plt.title("current")

#%%

plt.figure()
plt.plot(data1.IMURightShankAccelA/1000 - 5)
# plt.plot(data1.IMURightShankAccelB/1000)
# plt.plot(data1.IMURightShankAccelC/1000)
# plt.plot(R_Leg)
# plt.plot(L_Leg)
plt.title("Accel A")

#%%
plt.figure()
# plt.plot(data1.IMURightShankAccelA/1000)
plt.plot(data1.IMURightShankAccelB/1000)
# plt.plot(data1.IMURightShankAccelC/1000)
# plt.plot(R_Leg)
# plt.plot(L_Leg)
plt.title("Accel B")

#%%


plt.figure()
# plt.plot(data1.IMURightShankAccelA/1000)
# plt.plot(data1.IMURightShankAccelB/1000)
plt.plot(data1.IMURightShankAccelC/1000)
plt.plot(R_Leg)
plt.plot(L_Leg)
plt.title("Accel C")


#%%
plt.figure()
# plt.plot(data1.IMURightShankAccelA/1000)
# plt.plot(data1.IMURightShankAccelB/1000)
plt.plot(data1.IMURightThighAccelC/1000)
plt.plot(R_Leg)
plt.plot(L_Leg)
plt.title("Accel C thigh")

#%%
plt.figure()
plt.plot(data1.IMURightShankMagA)
plt.plot(data1.IMURightShankMagB)
plt.plot(data1.IMURightShankMagC)

#%%

# plt.figure()
# plt.plot(data1.IMURightShankGyroA/1000)
# plt.plot(data1.IMURightShankGyroB/1000)
# plt.plot(data1.IMURightShankGyroC/1000)
# plt.plot(R_Leg)
# plt.plot(L_Leg)
# plt.plot(Mode)

#%%
# plt.figure()
# plt.plot(data1.IMURightThighGyroA/1000)
# plt.plot(data1.IMURightThighGyroB/1000)
# plt.plot(data1.IMURightThighGyroC/1000)

#%%

# plt.figure()
# # plt.plot(data1.IMURightThighGyroA/1000)
# plt.plot(data1.IMURightThighAccelC/1000)
# plt.plot(data1.IMURightThighGyroC/1000)
# plt.plot(Mode)

#%%