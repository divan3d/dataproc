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

# pas oublier left thigh
def segmentation(data,labelsMoCap):    
     
    #get indices for shank
    ishank1 = labelsMoCap.index('RightShank1')
    ishank2 = labelsMoCap.index('RightShank2')
    ishank3 = labelsMoCap.index('RightShank3')
    ishank4 = labelsMoCap.index('RightShank4')
    ishank5 = labelsMoCap.index('RightShank5')
    
    #get inices for thigh
    ithigh1 = labelsMoCap.index('RightThigh1')
    ithigh2 = labelsMoCap.index('RightThigh2')
    ithigh3 = labelsMoCap.index('RightThigh3')
    ithigh4 = labelsMoCap.index('RightThigh4')
    
    #get inidces for shoulders
    ishoulders1 = labelsMoCap.index('Shoulders1')
    ishoulders2 = labelsMoCap.index('Shoulders2')
    ishoulders3 = labelsMoCap.index('Shoulders3')
    
    #get indices for lower KMA
    ilKMA1 = labelsMoCap.index('RightKMAL1')
    ilKMA2 = labelsMoCap.index('RightKMAL2')
    ilKMA3 = labelsMoCap.index('RightKMAL3')
    ilKMA4 = labelsMoCap.index('RightKMAL4')
    
    #get indices for upper KMA
    iuKMA1 = labelsMoCap.index('RightKMAU1')
    iuKMA2 = labelsMoCap.index('RightKMAU2')
    iuKMA3 = labelsMoCap.index('RightKMAU3')
    iuKMA4 = labelsMoCap.index('RightKMAU4')
    
    #get indices for TDU
    iTDU1 = labelsMoCap.index('Backpack1')
    iTDU2 = labelsMoCap.index('Backpack2')
    iTDU3 = labelsMoCap.index('Backpack3')
    iTDU4 = labelsMoCap.index('Backpack4')
    
    #initialize empty vectors
    shank = [0]*len(data)
    thigh = [0]*len(data)
    shoulders = [0]*len(data)
    KMAl  = [0]*len(data)
    KMAu  = [0]*len(data)
    TDU   = [0]*len(data)
    
    #fill empty vectors with corresponding marker positions
    for i in range(0,len(data)):
        shank[i]=np.row_stack((data[i][[ishank1,ishank2,ishank3, ishank4, ishank5],:]))
        thigh[i]=np.row_stack((data[i][[ithigh1,ithigh2,ithigh3,ithigh4],:]))
        shoulders[i]  =np.row_stack((data[i][[ishoulders1,ishoulders2,ishoulders3],:]))
        KMAl[i] =np.row_stack((data[i][[ilKMA1,ilKMA2,ilKMA3, ilKMA4],:]))
        KMAu[i] =np.row_stack((data[i][[iuKMA1,iuKMA2,iuKMA3, iuKMA4],:]))
        TDU[i]  =np.row_stack((data[i][[iTDU1,iTDU2,iTDU3, iTDU4],:]))
    
        
    return [shank,thigh,KMAl,KMAu,TDU,shoulders]

