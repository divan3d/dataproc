'''
reads out the label names of motion capture markers and groups them into segments
modified from marc's code
'''
import numpy as np


def generate_data(procObj):
          
    #initial positions.
    start_positions = np.array(procObj.points_array[0,:,0:3])

    #fill data and time vector with an initial entry
    data = [start_positions]
    time = np.zeros(len(procObj.points_array))
    time[0] = procObj.TIME_VECTOR[0]
    
    #iterate through all entries and fill time and data vector
    for iteration in range(1,len(procObj.points_array)):
        new_positions = np.array(procObj.points_array[iteration,:,0:3])
        data.append(new_positions)
        time[iteration] = procObj.TIME_VECTOR[iteration]
    
    #red out label names
    labels = procObj.labels

    return [data,labels,time]


def segmentation(data,labelsMoCap):    
     
    #get indices for shank
    ishank1 = labelsMoCap.index('Shank1')
    ishank2 = labelsMoCap.index('Shank2')
    ishank3 = labelsMoCap.index('Shank3')
    
    #get inices for thigh
    ithigh1 = labelsMoCap.index('Thigh1')
    ithigh2 = labelsMoCap.index('Thigh2')
    ithigh3 = labelsMoCap.index('Thigh3')
    ithigh4 = labelsMoCap.index('Thigh4')
    ithigh5 = labelsMoCap.index('Thigh5')
    
    #get inidces for hip
    ihip1 = labelsMoCap.index('Hip1')
    ihip2 = labelsMoCap.index('Hip2')
    ihip3 = labelsMoCap.index('Hip3')
    ihip4 = labelsMoCap.index('Hip4')
    
    #get indices for lower KMA
    ilKMA1 = labelsMoCap.index('KMAlow1')
    ilKMA2 = labelsMoCap.index('KMAlow2')
    ilKMA3 = labelsMoCap.index('KMAlow3')
    
    #get indices for upper KMA
    iuKMA1 = labelsMoCap.index('KMAup1')
    iuKMA2 = labelsMoCap.index('KMAup2')
    iuKMA3 = labelsMoCap.index('KMAup3')
    
    #get indices for TDU
    iTDU1 = labelsMoCap.index('TDU1')
    iTDU2 = labelsMoCap.index('TDU2')
    iTDU3 = labelsMoCap.index('TDU3')
    
    #initialize empty vectors
    shank = [0]*len(data)
    thigh = [0]*len(data)
    hip   = [0]*len(data)
    KMAl  = [0]*len(data)
    KMAu  = [0]*len(data)
    TDU   = [0]*len(data)
    
    #fill empty vectors with corresponding marker positions
    for i in range(0,len(data)):
        shank[i]=np.row_stack((data[i][[ishank1,ishank2,ishank3],:]))
        thigh[i]=np.row_stack((data[i][[ithigh1,ithigh2,ithigh3,ithigh4,ithigh5],:]))
        hip[i]  =np.row_stack((data[i][[ihip1,ihip2,ihip3,ihip4],:]))
        KMAl[i] =np.row_stack((data[i][[ilKMA1,ilKMA2,ilKMA3],:]))
        KMAu[i] =np.row_stack((data[i][[iuKMA1,iuKMA2,iuKMA3],:]))
        TDU[i]  =np.row_stack((data[i][[iTDU1,iTDU2,iTDU3],:]))
    
        
    return [shank,thigh,hip,KMAl,KMAu,TDU]

