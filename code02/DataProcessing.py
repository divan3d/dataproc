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
# vu que ça change le directory 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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
        os.chdir(r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\INTERMEDIATE")
        save_obj(obj, name )
    return
                
def save_all_bits_sep(dict_data, dirName):
    # save dict separately 
    for key in dict_data:
        tmp_name = dirName + "_" + key + ".pkl"
        save_obj(dict_data[key], tmp_name)
    return

def save_trials_sep(dirName, dict_data):  
    os.chdir(r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test")
            
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
tpathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MOCAP\2021_08_03\FL1.pkl"
tpathMS = r"E:\ETHZ\mast_sem_IV\pdm\experiment\MYOSUIT\2021_07_20\MS_log_SA_FL1.pkl"

# # enfaite il y a aussi la force 
test_dpcollect, seg = DataProcessingCollectNEW.data_processing_collect(tpathMoCap, tpathMS, False)

# # # Save 
tDPC_outname = "SB_030821_COL_FL1.pkl"
file_save(test_dpcollect, tDPC_outname, True, False)

#%% Interpret Mocap data - extract angles 

# Input data
tDI_filein = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\INTERMEDIATE\SB_030821_COL_FL1.pkl"
tDI_datain = op_pickle(tDI_filein)

# # old
# # test_pdint = DataProcessingMCInterpretation.data_processing_interpretation(tDI_datain)

# # new 
test_pdint = DataProcessingMCInterpretationNEW.data_processing_interpretation(tDI_datain, False)

# # Save
tDI_outname = "SB_030821_INT_FL1.pkl"
file_save(test_pdint, tDI_outname, True, True)



#%% remove offset 

tcut_filein = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\INTERMEDIATE\SB_030821_INT_FL1.pkl"
d_in = op_pickle(tcut_filein)
# # d_in = test_pdint

DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "shank")
DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "thigh")

# # effacer après 
# # tcut_filein = r"E:\ETHZ\mast_sem_IV\pdm\code02\SA_200721_FUN_00.pkl"
# # d_in = op_pickle(tcut_filein)

DataProcessingFunctions.plot_res_shank(d_in)
DataProcessingFunctions.plot_res_thigh(d_in)

# Save
ro_file_name = "SB_030821_FUN_FL1.pkl"
file_save(d_in, ro_file_name, True, True)


#%% separate trials (need MS data)

# file1 = r"E:\ETHZ\mast_sem_IV\pdm\extracted_data\test\INTERMEDIATE\SA_200721_FUN_FL3_MS.pkl"
# # file1c = r"S01_cut_Mocap_angles_res"
# data1 = op_pickle(file1)
# dict_of_trials = DataProcessingFunctions.separate_trial(data1)

# save_trials_sep("SA_FL3_TEST", dict_of_trials)





