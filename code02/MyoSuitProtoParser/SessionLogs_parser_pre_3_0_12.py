# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 08:20:17 2020

@author: Gleb Koginov
"""

import numpy as np
import pandas as pd
import time

######


class log_parser():
    
    def __init__(self, PSessionLogs_pb2, log_path=None):
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
        
        ### PARSE THE LOG BODY ###
        varint = True
        buffer_message = bytearray()
        buffer_tag = []      
        
        double_varint = 0
        
        self.list_message_stream = [] # List of dictionaries 
            
        while self.counter < len(self.binary_log_data):
            
            if varint:
                
                varint_tag = self.binary_log_data[self.counter]
                        
                # Field numbers >16 can take 2 bytes to encode #
                if varint_tag & 128:
                    double_varint = 1
                    self.counter += 1
                    if self.counter >= len(self.binary_log_data):
                        break
                    field_number = ((self.binary_log_data[self.counter] & 127) << 7 | (varint_tag & 127)) >> 3
                    buffer_message.append(varint_tag)
                    varint_tag = self.binary_log_data[self.counter]
                    
                else:
                    double_varint = 0
                    field_number = (varint_tag >> 3)
                    field_wire = varint_tag & 7
                
                # If field already exists it's a new message #
                if field_number in buffer_tag:
                    
                    if double_varint:
                        self.logger.ParseFromString(buffer_message[:-1])
                    else:
                        self.logger.ParseFromString(buffer_message)
                    dict_message_parsed = {} # Clear the memory reference #
        
                    for keys in self.field_names:
                        
                        dict_message_parsed[keys] = getattr(self.logger, keys)
                        
                    self.list_message_stream.append(dict_message_parsed)
                    
                    # Reset the variables #
                    buffer_tag = []
                    if double_varint:
                        self.counter -= 1
                        # temp_tag_value = buffer_message[-1]
                        buffer_message = bytearray()
                        # buffer_message.append(temp_tag_value)
                    else:
                        buffer_message = bytearray()
                    self.logger.Clear()
                    continue
                    # break
                else:
                    buffer_tag.append(field_number)
                
                buffer_message.append(varint_tag)
                varint = False
                self.counter += 1
                
            else:
                
                    
                byte_read = self.binary_log_data[self.counter]
                buffer_message.append(byte_read)
                
                while self.binary_log_data[self.counter] & 128:   
                    self.counter += 1
                    if self.counter >= len(self.binary_log_data):
                        break
                    byte_read = self.binary_log_data[self.counter]
                    buffer_message.append(byte_read)
                
                self.counter += 1
                varint = True
                
    def create_dataframe(self):
        
        df_result_stream = pd.DataFrame(self.list_message_stream)
        df_result_stream = df_result_stream.set_index("DataTimestamp")
        
        return df_result_stream



if __name__ == "__main__":
    from python_protos import PSessionLogs_pb2
    
    start_time = time.time()
    log_path = r'C:\Users\CAD Machine 2\Downloads\SessionLogs_DC6215C6-4559-4637-8C8B-C0180D702559.myo'
    
    data_log = log_parser(PSessionLogs_pb2, log_path=log_path)
    data_log.process_log()
    
    results = data_log.create_dataframe()
    
    
    print("Parsing took: {}[s]".format(np.round(time.time() - start_time),1))