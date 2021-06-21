# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:42:10 2020

@author: Gleb Koginov
"""
import numpy as np

class data_integrity():
    def __init__(self):
        self.gyro_var_tresh = 1000
        self.status_log = {}
        
    def run_check(self, data_log):
        
        self.status_log = {"RT_GyrC": 1,
                            "LT_GyrC": 1,
                            "RS_GyrC": 1,
                            "LS_GyrC": 1,
                            "Tr_GyrC": 1}
            
        self.check_gyro(data_log)
        failed_ids = np.array([i for i, e in enumerate(list(self.status_log.values())) if e == 0])
        failed_chans = np.array(list(self.status_log.keys()))[failed_ids.astype(int)]
        return(failed_chans)
        
    def check_gyro(self, data_log):
        chan_names = ["RT_GyrC", "LT_GyrC", "RS_GyrC", "LS_GyrC", "Tr_GyrC"]
        for chans in chan_names:
            if np.var(data_log[chans]) < 1000:
                self.status_log[chans] = 0
                
                
class data_integrity_legacy():
    def __init__(self):
        self.gyro_var_tresh = 1000
        self.status_log = {}
        
    def run_check(self, data_log):
        
        self.status_log = {"RT_Gyr": 1,
                           "LT_Gyr": 1,
                           "RS_Gyr": 1,
                           "LS_Gyr": 1,
                           "Tr_Gyr": 1}
            
        self.check_gyro(data_log)
        failed_ids = np.array([i for i, e in enumerate(list(self.status_log.values())) if e == 0])
        failed_chans = np.array(list(self.status_log.keys()))[failed_ids.astype(int)]
        return(failed_chans)
        
    def check_gyro(self, data_log):
        chan_names = ["RT_Gyr", "LT_Gyr", "RS_Gyr", "LS_Gyr", "Tr_Gyr"]
        for chans in chan_names:
            if np.var(data_log[chans]) < 1000:
                self.status_log[chans] = 0