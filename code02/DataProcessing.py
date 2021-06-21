# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:05:38 2021

@author: ksdiv

prepares dataframes/ sets for HMM 
"""


import pickle
import DataProcessingMCInterpretation
import DataProcessingFunctions, DataProcessingCollect 
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
                

# DataProcessingCollect -> 
tpathMoCap = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MoCap_Logs\S01_19112020\S01_19112020_Trial02.pkl"
tpathForce = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\ForceCell_Logs\S01_19112020_Force\S01_19112020_T02.csv"
tpathMS    = r"E:\ETHZ\mast_sem_IV\pdm\marc\03_Relevant Files\03_Software\MyoSuit_Logs\S01_19112020_MS\S01_19112020_T02.pickle"

tDPC_outname = "T_DPC_E.pkl"

test_dpcollect = DataProcessingCollect.data_processing_collect(tpathMoCap, tpathForce, tpathMS, tDPC_outname)

file_save(test_dpcollect, tDPC_outname, True, False)


# DataProcessingMCInterpretation -> 
tDI_filein = r"E:\ETHZ\mast_sem_IV\pdm\code02\T_DPC.pkl"
tDI_datain = op_pickle(tDI_filein)
tDI_outname = "T_INT_E.pkl"

test_pdint = DataProcessingMCInterpretation.data_processing_interpretation(tDI_datain)

file_save(test_pdint, tDI_outname, True, True)

# pr tester si mocap joue bien, peut s'arreter ici ? ou s'arrete apres avoir enlever le offset 
# peut etre mettre une fonction plot, juste histoire de vérifier à l'oeil, limite comparer à
# données qu'on a déjà 


# if dynamic 
# separate into gait cycles, remove offset, 

tcut_filein = r"E:\ETHZ\mast_sem_IV\pdm\code02\T_INT.pkl"
cut_file_name = "T_CUT_E.pkl"
# d_in = op_pickle(tcut_filein)
d_in = test_pdint

DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "shank")
DataProcessingFunctions.dyn_remove_offset_bodypart(d_in, "thigh")

cut_dict = DataProcessingFunctions.dyn_cut_to_ind_gait_cycle(d_in)

file_save(cut_dict, cut_file_name, True, False)

#cut further, clean 

# if static 
# keep only pos1, remove offset, value for behavior near small peaks (à explorer)