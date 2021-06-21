# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:47:22 2020

@author: Gleb Koginov
"""

import time
import pandas as pd
import numpy as np

class log_parser():
    
    def __init__(self, PSessionLogs_pb2, log_path=None):
        self.PSessionLogs_pb2 = PSessionLogs_pb2
        self.logger = PSessionLogs_pb2.PSessionLogsDetail() # log detail message object 
        self.field_names = [field.name for field in self.logger.DESCRIPTOR.fields]
    
        # Define the header structure (dictionary) #
        self.dict_log_header = {}
        self.dict_log_header['EntityType'] = -1
        self.dict_log_header['NexusProtocol'] = -1
        self.dict_log_header['GUID'] = -1
    
        
        
        self.log_path = log_path
        
        
    def read_data(self):        
        with open(self.log_path, 'rb') as f:
            self.binary_log_data = f.read()
        f.close()
        
    def process_log(self, binary_data = None):
        self.counter = 0
        if self.log_path:
            print("Reading data...")
            self.read_data()
        elif binary_data:
            self.binary_log_data = binary_data
        else:
            raise Exception("No file path or log data passed")

        print("Parsing log header...")
        self.parse_header()
        
        print("Parsing log body...")
        self.parse_log()
    
    def parse_header(self):

        ### PARSE THE HEADER ###
        # TODO: Change .proto to parse this automatically #
            
        varint = True
        buffer_message = []
        buffer_tag = []
        while True:
            
            buffer_message = []
            
            # Read the tag #
            if varint:
                
                buffer_message.append(self.binary_log_data[self.counter])
                varint_tag = buffer_message[0]
                field_number = (varint_tag >> 3) & 255 # Check the tag number 
                field_wire = varint_tag & 7 # Check the wire type
                
                # If this field already exists, header is finished #
                if field_number in buffer_tag:
                    break
                else:
                    buffer_tag.append(field_number)
                
                varint = False
                self.counter += 1
            
            # Read the message #
            else:
                
                # Entity type (int32) #
                if field_number == 1:
                    
                    byte_read = self.binary_log_data[self.counter]
                    buffer_message.insert(0, byte_read)            
                    self.counter += 1
                    
                    while self.binary_log_data[self.counter] & 128: # Check MSB to see if next byte also used 
                        byte_read = self.binary_log_data[self.counter]
                        buffer_message.insert(0, byte_read) # Prepend if yes
                        self.counter += 1
                    
                    # Build value #
                    for j in range(0, len(buffer_message)):
                        buffer_message[j] = buffer_message[j] << 8 * (len(buffer_message) - 1)                
                       
                    
                    self.dict_log_header['EntityType'] = np.sum(buffer_message)
                    
                # Nexus protocol version (str) #
                if field_number == 2:
                    length_string = self.binary_log_data[self.counter] # Incomming string length #
                    self.counter += 1
                    self.dict_log_header['NexusProtocol'] = self.binary_log_data[self.counter:self.counter+length_string]
                    self.counter += length_string
                
                # GUID (str) #
                if field_number == 3:
                    length_string = self.binary_log_data[self.counter]
                    self.counter += 1
                    self.dict_log_header['GUID'] = self.binary_log_data[self.counter:self.counter+length_string]
                    self.counter += length_string
                    break
                varint = True
                
    def parse_log(self):

        self.list_message_stream = []
        self.msg_length_list = []
        dataMsg= []
        msgSer = self.binary_log_data
        
        while self.counter < len(msgSer):
            
            # if len(self.list_message_stream) == 48:
            #     print("here")
            
            tagVarint = msgSer[self.counter]
                    
            # Field numbers >16 can take 2 bytes to encode #
            if tagVarint & 128:
                self.counter += 1
                if self.counter >= len(msgSer):
                    break
                fldN = ((msgSer[self.counter] & 127) << 7 | (tagVarint & 127)) >> 3
                dataMsg.append(tagVarint)
                tagVarint = msgSer[self.counter]
                dataMsg.append(tagVarint)
                self.counter += 1
            else: 
                print("Wrong tagVarint detected")
                break
                
            # Get the message #
            try:
                list_length = [msgSer[self.counter]]
            except Exception as e:
                print("Issue during processing - last message may be missing")
                self.counter += 1
                continue
            while msgSer[self.counter] & 128:
                self.counter += 1 
                list_length.append(msgSer[self.counter])
                
            msg_length = 0
            for k in range(len(list_length), 0, -1):
                msg_length = ((list_length[k-1] & 127) << 7 *(k-1)) | msg_length
            
            # msg_length = ((msgSer[self.counter] & 127) << 7) | (msgSer[self.counter-1] & 127)
            self.msg_length_list.append(msg_length)
            self.counter += 1
            self.list_message_stream.append(msgSer[self.counter:self.counter + msg_length])
            self.counter += msg_length
            
        
        
        listOfObj = [self.PSessionLogs_pb2.PSessionLogsDetail() for i in range(0, len(self.list_message_stream)-1)]
        
        for i in range(0, len(listOfObj)):
            listOfObj[i].ParseFromString(self.list_message_stream[i])


        self.resultDF = pd.DataFrame([{i:getattr(listOfObj[j], i) for i in self.field_names} for j in range(0, len(listOfObj))])



if __name__ == "__main__":
    from python_protos import PSessionLogs_pb2
    
    start_time = time.time()
    log_path = r'D:\static-sitting.myo'
    
    data_log = log_parser(PSessionLogs_pb2, log_path=log_path)
    data_log.process_log()