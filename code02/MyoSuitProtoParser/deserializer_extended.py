# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:16:21 2020

@author: CAD Machine 2
"""

from python_protos.v_1_1_0 import PSessionHeader_pb2
from python_protos.v_1_1_0 import PSessionLogs_pb2
from python_protos.v_1_1_0 import PGenConfig_pb2
from python_protos.v_1_1_0 import PPatientProfile_pb2
import SessionLogs_parser
import pickle
from datetime import datetime

startTime = datetime.now()
#############################
fld = r"C:\Users\wirth\polybox\MSc\Master Thesis\03_Relevant Files\03_Software\MyoSuit_Logs\S05_23112020_MS" # Folder where to look
fl2Use = "\S05_23112020_T02.myo" # File to decode


#############################
# 0 - header
# 1 - log
# 2 - patient profile
# 3 - general config
protobuf_to_process = 1


#############################

overwrite_file = 0 # set to 1 if you made changes and want to overwrite the original file

#############################

save_pickle = 1

#############################
###### SESSION HEADER ######
if protobuf_to_process == 0:
    with open(fld + fl2Use, 'rb') as f:
        read_metric =  PSessionHeader_pb2.PSessionHeader()
        data_input = f.read()
        while data_input.endswith(b"\x00"):
            data_input = data_input[:-1]
        read_metric.ParseFromString(data_input)
    
###### SESSION LOG ######
if protobuf_to_process == 1:
    with open(fld + fl2Use, 'rb') as f:
        data_input = f.read()
    parser_log = SessionLogs_parser.log_parser(PSessionLogs_pb2)
    parser_log.process_log(binary_data = data_input)
    read_metric = parser_log.resultDF
    
##### PATIENT PROFILE ######
if protobuf_to_process == 2:
    with open(fld + fl2Use, 'rb') as f:
        read_metric =  PPatientProfile_pb2.PPatientProfile()
        data_input = f.read()
        while data_input.endswith(b"\x00"):
            data_input = data_input[:-1]
        read_metric.ParseFromString(data_input)


###### GENERAL CONFIGURATION #####
if protobuf_to_process == 3:
    with open(fld + fl2Use, 'rb') as f:
        read_metric =  PGenConfig_pb2.GeneralConfiguration()
        data_input = f.read()
        while data_input.endswith(b"\x00"):
            data_input = data_input[:-1]
        read_metric.ParseFromString(data_input)

###### SAVE THE .pickle FILE #####
if save_pickle == 1:
    with open(fld + fl2Use.replace(".myo", "_extended.pickle"), 'wb') as handle:
        pickle.dump(read_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
###### SAVE THE .MYO FILE #####
if overwrite_file == 1:
    value = input("Type 1 to save the file: ")
    if value == '1':
        f = open(fld + fl2Use, "wb")
        f.write(read_metric.SerializeToString())
        f.close()
        print("File Saved!")
    else: 
        print("File was not saved!")

print(datetime.now() - startTime)