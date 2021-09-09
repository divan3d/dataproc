# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:39:14 2021

@author: Gleb
"""
import pandas as pd
import pickle
import os 
from matplotlib import pyplot as plt

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

headers = ["tv_sec","tv_nsec","TDUTimestamp","IMURightShankAccelA","IMURightShankAccelB","IMURightShankAccelC","IMURightShankGyroA","IMURightShankGyroB","IMURightShankGyroC","IMURightShankMagA","IMURightShankMagB","IMURightShankMagC","IMURightShankKalmanAngle","IMULeftShankAccelA","IMULeftShankAccelB","IMULeftShankAccelC","IMULeftShankGyroA","IMULeftShankGyroB","IMULeftShankGyroC","IMULeftShankMagA","IMULeftShankMagB","IMULeftShankMagC","IMULeftShankKalmanAngle","IMURightThighAccelA","IMURightThighAccelB","IMURightThighAccelC","IMURightThighGyroA","IMURightThighGyroB","IMURightThighGyroC","IMURightThighMagA","IMURightThighMagB","IMURightThighMagC","IMURightThighKalmanAngle","IMULeftThighAccelA","IMULeftThighAccelB","IMULeftThighAccelC","IMULeftThighGyroA","IMULeftThighGyroB","IMULeftThighGyroC","IMULeftThighMagA","IMULeftThighMagB","IMULeftThighMagC","IMULeftThighKalmanAngle","IMUTrunkAccelA","IMUTrunkAccelB","IMUTrunkAccelC","IMUTrunkGyroA","IMUTrunkGyroB","IMUTrunkGyroC","IMUTrunkMagA","IMUTrunkMagB","IMUTrunkMagC","IMUTrunkKalmanAngle","HallRightMotor","HallLeftMotor","ActualCurrentRightMotor","ActualCurrentLeftMotor","CurrentRightMotor","CurrentLeftMotor","TempRightMotor","TempLeftMotor","sensorS_BatteryVolt","sensorS_StateAlarmState","sensorS_StateSysInf","sensorS_StateLeftStanceDetection","sensorS_StateRightStanceDetection","sensorS_StateLeftForce","sensorS_StateRightForce","sensorS_StateSysMode","sensorS_CommandRightLegCalibrationLevel","sensorS_CommandLeftLegCalibrationLevel","sensorS_StateBodyPosition","sensorS_Dummy0","sensorS_Dummy1","commandS_lxEnable","commandS_rxEnable","commandS_lxCalibration","commandS_rxCalibration","commandS_lxCurrent","commandS_rxCurrent","extSensorS_Analog0",   "extSensorS_Analog1",   "extSensorS_Analog2",   "extSensorS_Analog3","extSensorS_SyncIn","extSensorS_UserButton", "extSensorS_MarkerGUI","spiDiagnosis_SPISuccess", "spiDiagnosis_SPIFail"]
log = pd.read_csv(r"E:/ETHZ/mast_sem_IV/pdm/experiment/MYOSUIT/SA/MS_log_SA_FL1_2.txt", skiprows=[0, 1, 2, 3, 4, 5], skipfooter=7)
log.columns = headers

# plt.figure()
# plt.plot(log["TDUTimestamp"].iloc[1:])
# print("time of the run: {}".format((log["TDUTimestamp"].values[-1] - log["TDUTimestamp"].values[1]) / 1000 /60))
# plt.figure()
# plt.plot(log["IMULeftShankGyroC"].iloc[1:])


name_file = "MS_log_SA_FL1_2.pkl"
os.chdir(r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\SA")
save_obj(log, name_file)

#%%

plt.figure()
plt.plot(log["sensorS_StateRightForce"].iloc[1:])

plt.figure()
plt.plot(log["IMULeftShankGyroC"].iloc[1:])

mode = log["sensorS_StateSysMode"]
sysinf = log["sensorS_StateSysInf"]
sysinf = sysinf - 8
Mode = mode * sysinf
Mode = Mode.replace(0,3)

sync = log["extSensorS_SyncIn"]

plt.figure()
plt.plot(Mode)
plt.plot(sync)