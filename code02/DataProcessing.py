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
import DataProcessingFunctions, dataprocessing_nv
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
    #pickkle_file que peut etre True si c'est un dataframe 
    if pickle_file:
        obj.to_pickle(name)
    
    if save_file :
        os.chdir(dname)
        save_obj(obj, name )
    return
                
#%% Retrieve raw data, adjust to same timeline, correct mocap etc

# old 

# Input data 
# tpathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MoCap_Logs\S06_23112020\S06_23112020_T02.pkl"
# tpathForce = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\ForceCell_Logs\S06_23112020_Force\S06_23112020_T02.csv"
# tpathMS    = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MyoSuit_Logs\S06_23112020_MS\S06_23112020_T02.pickle"

# test_dpcollect = DataProcessingCollect.data_processing_collect(tpathMoCap, tpathForce, tpathMS, tDPC_outname)

# new 

# Input data 
# tpathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\experiment\2021_07_09_BALGRIST\Kanako Cal 02.pkl"
# tpathMS = "bla.pkl"

# test_dpcollect = dataprocessing_nv.data_processing_collect(tpathMoCap, tpathMS, tDPC_outname, False)

# Save 
# tDPC_outname = "new_test.pkl"
# file_save(test_dpcollect, tDPC_outname, True, False)

#%% Interpret Mocap data - extract angles 

# Input data
tDI_filein = r"E:\ETHZ\mast_sem_IV\pdm\code02\new_test.pkl"
tDI_datain = op_pickle(tDI_filein)

# old
# test_pdint = DataProcessingMCInterpretation.data_processing_interpretation(tDI_datain)

# new 
test_pdint = DataProcessingMCInterpretationNEW.data_processing_interpretation(tDI_datain, False)

# Save
tDI_outname = "new_test_mod_getdirec_int.pkl"
file_save(test_pdint, tDI_outname, True, True)

# # pr tester si mocap joue bien, peut s'arreter ici ? ou s'arrete apres avoir enlever le offset 
# # peut etre mettre une fonction plot, juste histoire de vérifier à l'oeil, limite comparer à
# # données qu'on a déjà 

#%% remove offset 

# tcut_filein = r"E:\ETHZ\mast_sem_IV\pdm\code02\new_test_shankdiff_int.pkl"
# d_in = op_pickle(tcut_filein)
d_in = test_pdint

DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "shank")
DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "thigh")

# Save
ro_file_name = "test_mod_getdirec_functions_not_cut.pkl"
file_save(d_in, ro_file_name, True, False)


