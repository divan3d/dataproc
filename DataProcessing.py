# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:05:38 2021

@author: ksdiv

prepares dataframes/ sets for ML
fonctionne que si est dans le fichier code02, faudra changer ? 
"""


import pickle
import DataProcessingMCInterpretationNEW
import DataProcessingMCInterpretation
import DataProcessingFunctions, DataProcessingCollectNEW
# import DataProcessingFunctions, DataProcessingCollect 
import os

# import DataProcessingCollect doit etre à la fin parce que vu que ça va
# importer plein d'autres fonctions ça arrive pas à importer d'autres fichier
# vu que ça change le directory -- normalement réparé 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% Naming

# change this
subject = "SC"
fl = "FL1"

# directory
dir_name_int = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject + "\INTERMEDIATE"
dir_name_sep = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\\" + subject

# input 
input_mocap = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MOCAP\\" + subject + "\MC_" + subject + "_" + fl  +".pkl"
input_myosuit = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\\" + subject + "\MS_log_" + subject + "_" + fl  +".pkl"

# created files 
file_collected = subject + "_COL_" + fl + ".pkl"
file_interpreted = subject + "_INT_" + fl + ".pkl"
file_function = subject + "_FUN_" + fl + ".pkl"

#%%

def op_pickle(file):
    with open(file,'rb') as fileopen:
        data=pickle.load(fileopen)
        return data

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def file_save(obj, name, save_file, pickle_file):
    #pickle_file que peut etre True si c'est un dataframe 
    if pickle_file:
        obj.to_pickle(name)
    
    if save_file :
        os.chdir(dir_name_int)
        save_obj(obj, name )
    return
                
def save_all_bits_sep(dict_data, dirName):
    # save dict separately 
    for key in dict_data:
        tmp_name = dirName + "_" + key + ".pkl"
        save_obj(dict_data[key], tmp_name)
    return

def save_trials_sep(dirName, dict_data):  
    os.chdir(dir_name_sep)
            
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")    
        
    os.chdir(dirName)
    # print("Current working directory: {0}".format(os.getcwd()))

    save_all_bits_sep(dict_data, dirName) 
    

#%% Retrieve raw data, adjust to same timeline, correct mocap etc

# old 

# # Input data 
# tpathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MoCap_Logs\S06_23112020\S06_23112020_T02.pkl"
# tpathForce = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\ForceCell_Logs\S06_23112020_Force\S06_23112020_T02.csv"
# tpathMS    = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MyoSuit_Logs\S06_23112020_MS\S06_23112020_T02.pickle"

# test_dpcollect = DataProcessingCollect.data_processing_collect(tpathMoCap, tpathForce, tpathMS, tDPC_outname)

# new 

# Input data 
tpathMoCap = input_mocap
tpathMS = input_myosuit

# # enfaite il y a aussi la force 
test_dpcollect, seg = DataProcessingCollectNEW.data_processing_collect(tpathMoCap, tpathMS, True)

# # # Save 
tDPC_outname = file_collected
file_save(test_dpcollect, tDPC_outname, True, False)

#%% Interpret Mocap data - extract angles 

# Input data
tDI_filein = dir_name_int + "\\" + file_collected
tDI_datain = op_pickle(tDI_filein)

# # old
# # test_pdint = DataProcessingMCInterpretation.data_processing_interpretation(tDI_datain)

# # new 
test_pdint = DataProcessingMCInterpretationNEW.data_processing_interpretation(tDI_datain, True)

# # Save
tDI_outname = file_interpreted
file_save(test_pdint, tDI_outname, True, True)



#%% remove offset 

tcut_filein = dir_name_int + "\\" + file_interpreted
d_in = op_pickle(tcut_filein)
# # d_in = test_pdint

DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "shank")
DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "thigh")

# SE_FL1 et SC_FL1 - si thigh est inversé 
# DataProcessingFunctions.se_fl1(d_in)


DataProcessingFunctions.plot_res_shank(d_in)
DataProcessingFunctions.plot_res_thigh(d_in)

# Save
ro_file_name = file_function
file_save(d_in, ro_file_name, True, True)

#%% separate trials (need MS data)

file1 = dir_name_int + "\\" + file_function
# file1c = r"S01_cut_Mocap_angles_res"
data1 = op_pickle(file1)
dict_of_trials = DataProcessingFunctions.separate_trial(data1)

save_trials_sep(subject + "_" + fl, dict_of_trials)
os.chdir(dname)







