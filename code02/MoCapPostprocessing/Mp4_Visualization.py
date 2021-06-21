# -*- coding: utf-8 -*-
"""
@author: Marc Wirth

makes an mp4 file of the movements of the Myosuit of body segments
"""

#import needed libraries
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pickle
import matplotlib.colors as mcolors

#function to load only the needed data
def extractdata(file1,file2):
    
    #open data file
    with open(file1,'rb') as fileopen:
        data=pickle.load(fileopen)
            
    #load needed parameters
    markerlabels = data[2]
    markercloud = data[3]
    segments = data[6]

    #load second file and needed data
    with open(file2,'rb') as fileopen:
        data2=pickle.load(fileopen)    
    refvecs = data2[28]
    
    return segments,markercloud,markerlabels,refvecs

#make data frame in the form needed for the animation
def buildanimdata(seg,markercloud,labels,displaytype):
    
    data = []
    
    #iterate over all frames
    for n in range(0,len(markercloud)):
        if displaytype == 'body':
            #fill the frame with the markers
            frame = np.zeros([27,3])
            
            frame[0,:] = markercloud[n,labels.index('Shank1'),0:3]
            frame[1,:] = markercloud[n,labels.index('Shank2'),0:3]
            frame[2,:] = markercloud[n,labels.index('Shank3'),0:3]
            
            frame[3,:] = seg[0][n][0,:]
            frame[4,:] = seg[0][n][1,:]
            frame[5,:] = seg[0][n][2,:]
           
            frame[6,:] = markercloud[n,labels.index('Thigh1'),0:3]
            frame[7,:] = markercloud[n,labels.index('Thigh2'),0:3]
            frame[8,:] = markercloud[n,labels.index('Thigh3'),0:3]
            frame[9,:] = markercloud[n,labels.index('Thigh4'),0:3]
            frame[10,:] = markercloud[n,labels.index('Thigh5'),0:3]
            
            frame[11,:] = seg[1][n][0,:]
            frame[12,:] = seg[1][n][1,:]
            frame[13,:] = seg[1][n][2,:]
            frame[14,:] = seg[1][n][3,:]
            frame[15,:] = seg[1][n][4,:]
            
            frame[16,:] = markercloud[n,labels.index('Hip1'),0:3]
            frame[17,:] = markercloud[n,labels.index('Hip2'),0:3]
            frame[18,:] = markercloud[n,labels.index('Hip3'),0:3]
            frame[19,:] = markercloud[n,labels.index('Hip4'),0:3]
            
            frame[20,:] = seg[2][n][0,:]
            frame[21,:] = seg[2][n][1,:]
            frame[22,:] = seg[2][n][2,:]
            frame[23,:] = seg[2][n][3,:]
            
            frame[24,:] = (seg[0][n][0,:]+seg[0][n][1,:]+seg[0][n][2,:])/3
            frame[25,:] = (seg[1][n][0,:]+seg[1][n][1,:]+seg[1][n][2,:]+seg[1][n][3,:]+seg[1][n][4,:])/5
            frame[26,:] = (seg[2][n][0,:]+seg[2][n][1,:]+seg[2][n][2,:]+seg[2][n][3,:])/4
            
        if displaytype == 'myosuit':
            #fill the frame with the markers
            frame = np.zeros([21,3])
            
            frame[0,:] = markercloud[n,labels.index('KMAlow1'),0:3]
            frame[1,:] = markercloud[n,labels.index('KMAlow2'),0:3]
            frame[2,:] = markercloud[n,labels.index('KMAlow3'),0:3]
            
            frame[3,:] = seg[3][n][0,:]
            frame[4,:] = seg[3][n][1,:]
            frame[5,:] = seg[3][n][2,:]
           
            frame[6,:] = markercloud[n,labels.index('KMAup1'),0:3]
            frame[7,:] = markercloud[n,labels.index('KMAup2'),0:3]
            frame[8,:] = markercloud[n,labels.index('KMAup3'),0:3]
            
            frame[9,:]  = seg[4][n][0,:]
            frame[10,:] = seg[4][n][1,:]
            frame[11,:] = seg[4][n][2,:]
            
            frame[12,:] = markercloud[n,labels.index('TDU1'),0:3]
            frame[13,:] = markercloud[n,labels.index('TDU2'),0:3]
            frame[14,:] = markercloud[n,labels.index('TDU3'),0:3]
            
            frame[15,:] = seg[5][n][0,:]
            frame[16,:] = seg[5][n][1,:]
            frame[17,:] = seg[5][n][2,:]
            
            frame[18,:] = (seg[3][n][0,:]+seg[3][n][1,:]+seg[3][n][2,:])/3
            frame[19,:] = (seg[4][n][0,:]+seg[4][n][1,:]+seg[4][n][2,:])/3
            frame[20,:] = (seg[5][n][0,:]+seg[5][n][1,:]+seg[5][n][2,:])/3
        
        #add frame to the data
        data.append(frame)
        
    return data
       
#function which returns the actual data points
def animate_scatters(iteration, zeroframe, data, scatters, displaytype):
    
    #frame number
    iteration = iteration + zeroframe
    print(iteration) 
    
    #offset the scatter points in the data frame
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    
    #calculate the center of masses of the segments
    if displaytype == 'body':
        CoMSh = (data[iteration][3,:] + data[iteration][4,:] + data[iteration][5,:])/3
        CoMTh = (data[iteration][11,:] + data[iteration][12,:] + data[iteration][13,:] + data[iteration][14,:] + data[iteration][15,:])/5
        CoMTr = (data[iteration][20,:] + data[iteration][21,:] + data[iteration][22,:] + data[iteration][23,:])/4
    if displaytype == 'myosuit':
        CoMKMAl = (data[iteration][3,:] + data[iteration][4,:] + data[iteration][5,:])/3
        CoMKMAu = (data[iteration][9,:] + data[iteration][10,:] + data[iteration][11,:])/3
        CoMTDU  = (data[iteration][15,:] + data[iteration][16,:] + data[iteration][17,:])/3
      
    #initialize global variables
    if displaytype == 'body':
        global Q1
        global Q2
        global Q3
        
        global QSh12
        global QSh13
        global QSh23
        
        global QTh12 
        global QTh13
        global QTh14 
        global QTh15 
        global QTh23 
        global QTh24 
        global QTh25 
        global QTh34 
        global QTh35 
        global QTh45 
        
        global QTr12
        global QTr13
        global QTr14
        global QTr23
        global QTr24
        global QTr34
        
    if displaytype == 'myosuit':
        global Q1
        global Q2
        global Q3
        
        global QKMAl12
        global QKMAl13
        global QKMAl23
        
        global QKMAu12
        global QKMAu13
        global QKMAu23
        
        global QTDU12
        global QTDU13
        global QTDU23
    
    #remove entries in the quiver lists
    if displaytype == 'body':
        Q1.remove()
        Q2.remove()
        Q3.remove()
        QSh12.remove()
        QSh13.remove()
        QSh23.remove()
        QTh12.remove()
        QTh13.remove()
        QTh14.remove()
        QTh15.remove()
        QTh23.remove()
        QTh24.remove()
        QTh25.remove()
        QTh34.remove()
        QTh35.remove()
        QTh45.remove()
        QTr12.remove()
        QTr13.remove()
        QTr14.remove()
        QTr23.remove()
        QTr24.remove()
        QTr34.remove()
    if displaytype == 'myosuit':
        Q1.remove()
        Q2.remove()
        Q3.remove()
        QKMAl12.remove()
        QKMAl13.remove()
        QKMAl23.remove()
        QKMAu12.remove()
        QKMAu13.remove()
        QKMAu23.remove()
        QTDU12.remove()
        QTDU13.remove()
        QTDU23.remove()

    #fill the quiver lists with the actual vectors
    if displaytype == 'body':
        Q1 = ax.quiver(CoMSh[0],CoMSh[1],CoMSh[2],refvecs[0][iteration][0,2],refvecs[0][iteration][1,2],refvecs[0][iteration][2,2],length=400,color='xkcd:black')
        Q2 = ax.quiver(CoMTh[0],CoMTh[1],CoMTh[2],refvecs[1][iteration][0,2],refvecs[1][iteration][1,2],refvecs[1][iteration][2,2],length=400,color='xkcd:black')
        Q3 = ax.quiver(CoMTr[0],CoMTr[1],CoMTr[2],refvecs[2][iteration][0,2],refvecs[2][iteration][1,2],refvecs[2][iteration][2,2],length=400,color='xkcd:black')
    
        QSh12 = ax.quiver(data[iteration][3,0],data[iteration][3,1],data[iteration][3,2],data[iteration][4,0]-data[iteration][3,0],data[iteration][4,1]-data[iteration][3,1],data[iteration][4,2]-data[iteration][3,2],arrow_length_ratio=0,color='xkcd:blue')
        QSh13 = ax.quiver(data[iteration][3,0],data[iteration][3,1],data[iteration][3,2],data[iteration][5,0]-data[iteration][3,0],data[iteration][5,1]-data[iteration][3,1],data[iteration][5,2]-data[iteration][3,2],arrow_length_ratio=0,color='xkcd:blue')
        QSh23 = ax.quiver(data[iteration][4,0],data[iteration][4,1],data[iteration][4,2],data[iteration][5,0]-data[iteration][4,0],data[iteration][5,1]-data[iteration][4,1],data[iteration][5,2]-data[iteration][4,2],arrow_length_ratio=0,color='xkcd:blue')
        
        QTh12 = ax.quiver(data[iteration][11,0],data[iteration][11,1],data[iteration][11,2],data[iteration][12,0]-data[iteration][11,0],data[iteration][12,1]-data[iteration][11,1],data[iteration][12,2]-data[iteration][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh13 = ax.quiver(data[iteration][11,0],data[iteration][11,1],data[iteration][11,2],data[iteration][13,0]-data[iteration][11,0],data[iteration][13,1]-data[iteration][11,1],data[iteration][13,2]-data[iteration][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh14 = ax.quiver(data[iteration][11,0],data[iteration][11,1],data[iteration][11,2],data[iteration][14,0]-data[iteration][11,0],data[iteration][14,1]-data[iteration][11,1],data[iteration][14,2]-data[iteration][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh15 = ax.quiver(data[iteration][11,0],data[iteration][11,1],data[iteration][11,2],data[iteration][15,0]-data[iteration][11,0],data[iteration][15,1]-data[iteration][11,1],data[iteration][15,2]-data[iteration][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh23 = ax.quiver(data[iteration][12,0],data[iteration][12,1],data[iteration][12,2],data[iteration][13,0]-data[iteration][12,0],data[iteration][13,1]-data[iteration][12,1],data[iteration][13,2]-data[iteration][12,2],arrow_length_ratio=0,color='xkcd:red')
        QTh24 = ax.quiver(data[iteration][12,0],data[iteration][12,1],data[iteration][12,2],data[iteration][14,0]-data[iteration][12,0],data[iteration][14,1]-data[iteration][12,1],data[iteration][14,2]-data[iteration][12,2],arrow_length_ratio=0,color='xkcd:red')
        QTh25 = ax.quiver(data[iteration][12,0],data[iteration][12,1],data[iteration][12,2],data[iteration][15,0]-data[iteration][12,0],data[iteration][15,1]-data[iteration][12,1],data[iteration][15,2]-data[iteration][12,2],arrow_length_ratio=0,color='xkcd:red')
        QTh34 = ax.quiver(data[iteration][13,0],data[iteration][13,1],data[iteration][13,2],data[iteration][14,0]-data[iteration][13,0],data[iteration][14,1]-data[iteration][13,1],data[iteration][14,2]-data[iteration][13,2],arrow_length_ratio=0,color='xkcd:red')
        QTh35 = ax.quiver(data[iteration][13,0],data[iteration][13,1],data[iteration][13,2],data[iteration][15,0]-data[iteration][13,0],data[iteration][15,1]-data[iteration][13,1],data[iteration][15,2]-data[iteration][13,2],arrow_length_ratio=0,color='xkcd:red')
        QTh45 = ax.quiver(data[iteration][14,0],data[iteration][14,1],data[iteration][14,2],data[iteration][15,0]-data[iteration][14,0],data[iteration][15,1]-data[iteration][14,1],data[iteration][15,2]-data[iteration][14,2],arrow_length_ratio=0,color='xkcd:red')
    
        QTr12 = ax.quiver(data[iteration][20,0],data[iteration][20,1],data[iteration][20,2],data[iteration][21,0]-data[iteration][20,0],data[iteration][21,1]-data[iteration][20,1],data[iteration][21,2]-data[iteration][20,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr13 = ax.quiver(data[iteration][20,0],data[iteration][20,1],data[iteration][20,2],data[iteration][22,0]-data[iteration][20,0],data[iteration][22,1]-data[iteration][20,1],data[iteration][22,2]-data[iteration][20,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr14 = ax.quiver(data[iteration][20,0],data[iteration][20,1],data[iteration][20,2],data[iteration][23,0]-data[iteration][20,0],data[iteration][23,1]-data[iteration][20,1],data[iteration][23,2]-data[iteration][20,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr23 = ax.quiver(data[iteration][21,0],data[iteration][21,1],data[iteration][21,2],data[iteration][22,0]-data[iteration][21,0],data[iteration][22,1]-data[iteration][21,1],data[iteration][22,2]-data[iteration][21,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr24 = ax.quiver(data[iteration][21,0],data[iteration][21,1],data[iteration][21,2],data[iteration][23,0]-data[iteration][21,0],data[iteration][23,1]-data[iteration][21,1],data[iteration][23,2]-data[iteration][21,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr34 = ax.quiver(data[iteration][22,0],data[iteration][22,1],data[iteration][22,2],data[iteration][23,0]-data[iteration][22,0],data[iteration][23,1]-data[iteration][22,1],data[iteration][23,2]-data[iteration][22,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        
    if displaytype == 'myosuit':
        Q1 = ax.quiver(CoMKMAl[0],CoMKMAl[1],CoMKMAl[2],refvecs[3][iteration][0,2],refvecs[3][iteration][1,2],refvecs[3][iteration][2,2],length=400,color='xkcd:black')
        Q2 = ax.quiver(CoMKMAu[0],CoMKMAu[1],CoMKMAu[2],refvecs[4][iteration][0,2],refvecs[4][iteration][1,2],refvecs[4][iteration][2,2],length=400,color='xkcd:black')
        Q3 = ax.quiver(CoMTDU[0],CoMTDU[1],CoMTDU[2],refvecs[5][iteration][0,2],refvecs[5][iteration][1,2],refvecs[5][iteration][2,2],length=400,color='xkcd:black')
    
        QKMAl12 = ax.quiver(data[iteration][3,0],data[iteration][3,1],data[iteration][3,2],data[iteration][4,0]-data[iteration][3,0],data[iteration][4,1]-data[iteration][3,1],data[iteration][4,2]-data[iteration][3,2],arrow_length_ratio=0,color='xkcd:cyan')
        QKMAl13 = ax.quiver(data[iteration][3,0],data[iteration][3,1],data[iteration][3,2],data[iteration][5,0]-data[iteration][3,0],data[iteration][5,1]-data[iteration][3,1],data[iteration][5,2]-data[iteration][3,2],arrow_length_ratio=0,color='xkcd:cyan')
        QKMAl23 = ax.quiver(data[iteration][4,0],data[iteration][4,1],data[iteration][4,2],data[iteration][5,0]-data[iteration][4,0],data[iteration][5,1]-data[iteration][4,1],data[iteration][5,2]-data[iteration][4,2],arrow_length_ratio=0,color='xkcd:cyan')
        
        QKMAu12 = ax.quiver(data[iteration][9,0],data[iteration][9,1],data[iteration][9,2],data[iteration][10,0]-data[iteration][9,0],data[iteration][10,1]-data[iteration][9,1],data[iteration][10,2]-data[iteration][9,2],arrow_length_ratio=0,color='xkcd:light pink')
        QKMAu13 = ax.quiver(data[iteration][9,0],data[iteration][9,1],data[iteration][9,2],data[iteration][11,0]-data[iteration][9,0],data[iteration][11,1]-data[iteration][9,1],data[iteration][11,2]-data[iteration][9,2],arrow_length_ratio=0,color='xkcd:light pink')
        QKMAu23 = ax.quiver(data[iteration][10,0],data[iteration][10,1],data[iteration][10,2],data[iteration][11,0]-data[iteration][10,0],data[iteration][11,1]-data[iteration][10,1],data[iteration][11,2]-data[iteration][10,2],arrow_length_ratio=0,color='xkcd:light pink')
        
        QTDU12 = ax.quiver(data[iteration][15,0],data[iteration][15,1],data[iteration][15,2],data[iteration][16,0]-data[iteration][15,0],data[iteration][16,1]-data[iteration][15,1],data[iteration][16,2]-data[iteration][15,2],arrow_length_ratio=0,color='xkcd:mint')
        QTDU13 = ax.quiver(data[iteration][15,0],data[iteration][15,1],data[iteration][15,2],data[iteration][17,0]-data[iteration][15,0],data[iteration][17,1]-data[iteration][15,1],data[iteration][17,2]-data[iteration][15,2],arrow_length_ratio=0,color='xkcd:mint')
        QTDU23 = ax.quiver(data[iteration][16,0],data[iteration][16,1],data[iteration][16,2],data[iteration][17,0]-data[iteration][16,0],data[iteration][17,1]-data[iteration][16,1],data[iteration][17,2]-data[iteration][16,2],arrow_length_ratio=0,color='xkcd:mint')

  

#function which does the animation
def anim(data,zeroframe,length,name,displaytype,save=True):
    
    # Attaching 3D axis to the figure
    global ax
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    #determine number of iterations
    if length == "Full":
        iterations = len(data) - zeroframe
    elif length != "Full":
        iterations = length
        
        
    #define colors, transparencies and sizes of displayed points
    if displaytype == 'body':
        col=3*['xkcd:darkblue']+3*['xkcd:blue']+5*['xkcd:crimson']+5*['xkcd:red']+4*['xkcd:green']+4*['xkcd:lightgreen']+3*['xkcd:black']
        alp = 3*[0.3]+3*[1]+5*[0.3]+5*[1]+4*[0.3]+4*[1]+3*[0.5]
        msize = 3*[35]+3*[10]+5*[35]+5*[10]+4*[35]+4*[10]+3*[45]
    if displaytype == 'myosuit':
        col=3*['xkcd:aqua']+3*['xkcd:cyan']+3*['xkcd:dark pink']+3*['xkcd:light pink']+3*['xkcd:kelly green']+3*['xkcd:mint']+3*['xkcd:black']
        alp = 3*[0.3]+3*[1]+3*[0.3]+3*[1]+3*[0.3]+3*[1]+3*[0.5]
        msize = 3*[35]+3*[10]+3*[35]+3*[10]+3*[35]+3*[10]+3*[45]
    
    #define a scatter frame to display in the 3D plot
    scatters = [ ax.scatter(data[zeroframe][i,0:1], data[zeroframe][i,1:2], data[zeroframe][i,2:3],c=col[i],alpha=alp[i],s=msize[i]) for i in range(data[zeroframe].shape[0])]
    
    #initialize global variables
    if displaytype == 'body':
        global Q1
        global Q2
        global Q3
        
        global QSh12
        global QSh13
        global QSh23
        
        global QTh12 
        global QTh13
        global QTh14 
        global QTh15 
        global QTh23 
        global QTh24 
        global QTh25 
        global QTh34 
        global QTh35 
        global QTh45 
        
        global QTr12
        global QTr13
        global QTr14
        global QTr23
        global QTr24
        global QTr34
        
    if displaytype == 'myosuit':
        global Q1
        global Q2
        global Q3
        
        global QKMAl12
        global QKMAl13
        global QKMAl23
        
        global QKMAu12
        global QKMAu13
        global QKMAu23
        
        global QTDU12
        global QTDU13
        global QTDU23
    
    #calculate center of masses of the segments
    if displaytype == 'body':
        CoMSh = (data[zeroframe][3,:] + data[zeroframe][4,:] + data[zeroframe][5,:])/3
        CoMTh = (data[zeroframe][11,:] + data[zeroframe][12,:] + data[zeroframe][13,:] + data[zeroframe][14,:] + data[zeroframe][15,:])/5
        CoMTr = (data[zeroframe][20,:] + data[zeroframe][21,:] + data[zeroframe][22,:] + data[zeroframe][23,:])/4
    if displaytype == 'myosuit':
        CoMKMAl = (data[zeroframe][3,:] + data[zeroframe][4,:] + data[zeroframe][5,:])/3
        CoMKMAu = (data[zeroframe][9,:] + data[zeroframe][10,:] + data[zeroframe][11,:])/3
        CoMTDU  = (data[zeroframe][15,:] + data[zeroframe][16,:] + data[zeroframe][17,:])/3
    
    #define lines between markers
    if displaytype == 'body':
        Q1 = ax.quiver(CoMSh[0],CoMSh[1],CoMSh[2],refvecs[0][zeroframe][0,2],refvecs[0][zeroframe][1,2],refvecs[0][zeroframe][2,2],length=400,color='xkcd:black')
        Q2 = ax.quiver(CoMTh[0],CoMTh[1],CoMTh[2],refvecs[1][zeroframe][0,2],refvecs[1][zeroframe][1,2],refvecs[1][zeroframe][2,2],length=400,color='xkcd:black')
        Q3 = ax.quiver(CoMTr[0],CoMTr[1],CoMTr[2],refvecs[2][zeroframe][0,2],refvecs[2][zeroframe][1,2],refvecs[2][zeroframe][2,2],length=400,color='xkcd:black')
    
        QSh12 = ax.quiver(data[zeroframe][3,0],data[zeroframe][3,1],data[zeroframe][3,2],data[zeroframe][4,0]-data[zeroframe][3,0],data[zeroframe][4,1]-data[zeroframe][3,1],data[zeroframe][4,2]-data[zeroframe][3,2],arrow_length_ratio=0,color='xkcd:blue')
        QSh13 = ax.quiver(data[zeroframe][3,0],data[zeroframe][3,1],data[zeroframe][3,2],data[zeroframe][5,0]-data[zeroframe][3,0],data[zeroframe][5,1]-data[zeroframe][3,1],data[zeroframe][5,2]-data[zeroframe][3,2],arrow_length_ratio=0,color='xkcd:blue')
        QSh23 = ax.quiver(data[zeroframe][4,0],data[zeroframe][4,1],data[zeroframe][4,2],data[zeroframe][5,0]-data[zeroframe][4,0],data[zeroframe][5,1]-data[zeroframe][4,1],data[zeroframe][5,2]-data[zeroframe][4,2],arrow_length_ratio=0,color='xkcd:blue')
        
        QTh12 = ax.quiver(data[zeroframe][11,0],data[zeroframe][11,1],data[zeroframe][11,2],data[zeroframe][12,0]-data[zeroframe][11,0],data[zeroframe][12,1]-data[zeroframe][11,1],data[zeroframe][12,2]-data[zeroframe][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh13 = ax.quiver(data[zeroframe][11,0],data[zeroframe][11,1],data[zeroframe][11,2],data[zeroframe][13,0]-data[zeroframe][11,0],data[zeroframe][13,1]-data[zeroframe][11,1],data[zeroframe][13,2]-data[zeroframe][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh14 = ax.quiver(data[zeroframe][11,0],data[zeroframe][11,1],data[zeroframe][11,2],data[zeroframe][14,0]-data[zeroframe][11,0],data[zeroframe][14,1]-data[zeroframe][11,1],data[zeroframe][14,2]-data[zeroframe][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh15 = ax.quiver(data[zeroframe][11,0],data[zeroframe][11,1],data[zeroframe][11,2],data[zeroframe][15,0]-data[zeroframe][11,0],data[zeroframe][15,1]-data[zeroframe][11,1],data[zeroframe][15,2]-data[zeroframe][11,2],arrow_length_ratio=0,color='xkcd:red')
        QTh23 = ax.quiver(data[zeroframe][12,0],data[zeroframe][12,1],data[zeroframe][12,2],data[zeroframe][13,0]-data[zeroframe][12,0],data[zeroframe][13,1]-data[zeroframe][12,1],data[zeroframe][13,2]-data[zeroframe][12,2],arrow_length_ratio=0,color='xkcd:red')
        QTh24 = ax.quiver(data[zeroframe][12,0],data[zeroframe][12,1],data[zeroframe][12,2],data[zeroframe][14,0]-data[zeroframe][12,0],data[zeroframe][14,1]-data[zeroframe][12,1],data[zeroframe][14,2]-data[zeroframe][12,2],arrow_length_ratio=0,color='xkcd:red')
        QTh25 = ax.quiver(data[zeroframe][12,0],data[zeroframe][12,1],data[zeroframe][12,2],data[zeroframe][15,0]-data[zeroframe][12,0],data[zeroframe][15,1]-data[zeroframe][12,1],data[zeroframe][15,2]-data[zeroframe][12,2],arrow_length_ratio=0,color='xkcd:red')
        QTh34 = ax.quiver(data[zeroframe][13,0],data[zeroframe][13,1],data[zeroframe][13,2],data[zeroframe][14,0]-data[zeroframe][13,0],data[zeroframe][14,1]-data[zeroframe][13,1],data[zeroframe][14,2]-data[zeroframe][13,2],arrow_length_ratio=0,color='xkcd:red')
        QTh35 = ax.quiver(data[zeroframe][13,0],data[zeroframe][13,1],data[zeroframe][13,2],data[zeroframe][15,0]-data[zeroframe][13,0],data[zeroframe][15,1]-data[zeroframe][13,1],data[zeroframe][15,2]-data[zeroframe][13,2],arrow_length_ratio=0,color='xkcd:red')
        QTh45 = ax.quiver(data[zeroframe][14,0],data[zeroframe][14,1],data[zeroframe][14,2],data[zeroframe][15,0]-data[zeroframe][14,0],data[zeroframe][15,1]-data[zeroframe][14,1],data[zeroframe][15,2]-data[zeroframe][14,2],arrow_length_ratio=0,color='xkcd:red')
    
        QTr12 = ax.quiver(data[zeroframe][20,0],data[zeroframe][20,1],data[zeroframe][20,2],data[zeroframe][21,0]-data[zeroframe][20,0],data[zeroframe][21,1]-data[zeroframe][20,1],data[zeroframe][21,2]-data[zeroframe][20,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr13 = ax.quiver(data[zeroframe][20,0],data[zeroframe][20,1],data[zeroframe][20,2],data[zeroframe][22,0]-data[zeroframe][20,0],data[zeroframe][22,1]-data[zeroframe][20,1],data[zeroframe][22,2]-data[zeroframe][20,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr14 = ax.quiver(data[zeroframe][20,0],data[zeroframe][20,1],data[zeroframe][20,2],data[zeroframe][23,0]-data[zeroframe][20,0],data[zeroframe][23,1]-data[zeroframe][20,1],data[zeroframe][23,2]-data[zeroframe][20,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr23 = ax.quiver(data[zeroframe][21,0],data[zeroframe][21,1],data[zeroframe][21,2],data[zeroframe][22,0]-data[zeroframe][21,0],data[zeroframe][22,1]-data[zeroframe][21,1],data[zeroframe][22,2]-data[zeroframe][21,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr24 = ax.quiver(data[zeroframe][21,0],data[zeroframe][21,1],data[zeroframe][21,2],data[zeroframe][23,0]-data[zeroframe][21,0],data[zeroframe][23,1]-data[zeroframe][21,1],data[zeroframe][23,2]-data[zeroframe][21,2],arrow_length_ratio=0,color='xkcd:lightgreen')
        QTr34 = ax.quiver(data[zeroframe][22,0],data[zeroframe][22,1],data[zeroframe][22,2],data[zeroframe][23,0]-data[zeroframe][22,0],data[zeroframe][23,1]-data[zeroframe][22,1],data[zeroframe][23,2]-data[zeroframe][22,2],arrow_length_ratio=0,color='xkcd:lightgreen')
    
    if displaytype == 'myosuit':
        Q1 = ax.quiver(CoMKMAl[0],CoMKMAl[1],CoMKMAl[2],refvecs[3][zeroframe][0,2],refvecs[3][zeroframe][1,2],refvecs[3][zeroframe][2,2],length=400,color='xkcd:black')
        Q2 = ax.quiver(CoMKMAu[0],CoMKMAu[1],CoMKMAu[2],refvecs[4][zeroframe][0,2],refvecs[4][zeroframe][1,2],refvecs[4][zeroframe][2,2],length=400,color='xkcd:black')
        Q3 = ax.quiver(CoMTDU[0],CoMTDU[1],CoMTDU[2],refvecs[5][zeroframe][0,2],refvecs[5][zeroframe][1,2],refvecs[5][zeroframe][2,2],length=400,color='xkcd:black')
    
        QKMAl12 = ax.quiver(data[zeroframe][3,0],data[zeroframe][3,1],data[zeroframe][3,2],data[zeroframe][4,0]-data[zeroframe][3,0],data[zeroframe][4,1]-data[zeroframe][3,1],data[zeroframe][4,2]-data[zeroframe][3,2],arrow_length_ratio=0,color='xkcd:cyan')
        QKMAl13 = ax.quiver(data[zeroframe][3,0],data[zeroframe][3,1],data[zeroframe][3,2],data[zeroframe][5,0]-data[zeroframe][3,0],data[zeroframe][5,1]-data[zeroframe][3,1],data[zeroframe][5,2]-data[zeroframe][3,2],arrow_length_ratio=0,color='xkcd:cyan')
        QKMAl23 = ax.quiver(data[zeroframe][4,0],data[zeroframe][4,1],data[zeroframe][4,2],data[zeroframe][5,0]-data[zeroframe][4,0],data[zeroframe][5,1]-data[zeroframe][4,1],data[zeroframe][5,2]-data[zeroframe][4,2],arrow_length_ratio=0,color='xkcd:cyan')
        
        QKMAu12 = ax.quiver(data[zeroframe][9,0],data[zeroframe][9,1],data[zeroframe][9,2],data[zeroframe][10,0]-data[zeroframe][9,0],data[zeroframe][10,1]-data[zeroframe][9,1],data[zeroframe][10,2]-data[zeroframe][9,2],arrow_length_ratio=0,color='xkcd:light pink')
        QKMAu13 = ax.quiver(data[zeroframe][9,0],data[zeroframe][9,1],data[zeroframe][9,2],data[zeroframe][11,0]-data[zeroframe][9,0],data[zeroframe][11,1]-data[zeroframe][9,1],data[zeroframe][11,2]-data[zeroframe][9,2],arrow_length_ratio=0,color='xkcd:light pink')
        QKMAu23 = ax.quiver(data[zeroframe][10,0],data[zeroframe][10,1],data[zeroframe][10,2],data[zeroframe][11,0]-data[zeroframe][10,0],data[zeroframe][11,1]-data[zeroframe][10,1],data[zeroframe][11,2]-data[zeroframe][10,2],arrow_length_ratio=0,color='xkcd:light pink')
        
        QTDU12 = ax.quiver(data[zeroframe][15,0],data[zeroframe][15,1],data[zeroframe][15,2],data[zeroframe][16,0]-data[zeroframe][15,0],data[zeroframe][16,1]-data[zeroframe][15,1],data[zeroframe][16,2]-data[zeroframe][15,2],arrow_length_ratio=0,color='xkcd:mint')
        QTDU13 = ax.quiver(data[zeroframe][15,0],data[zeroframe][15,1],data[zeroframe][15,2],data[zeroframe][17,0]-data[zeroframe][15,0],data[zeroframe][17,1]-data[zeroframe][15,1],data[zeroframe][17,2]-data[zeroframe][15,2],arrow_length_ratio=0,color='xkcd:mint')
        QTDU23 = ax.quiver(data[zeroframe][16,0],data[zeroframe][16,1],data[zeroframe][16,2],data[zeroframe][17,0]-data[zeroframe][16,0],data[zeroframe][17,1]-data[zeroframe][16,1],data[zeroframe][17,2]-data[zeroframe][16,2],arrow_length_ratio=0,color='xkcd:mint')
  

    # Setting the axes properties
    ax.set_xlim3d([-500, 500])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1000, 1000])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-100, 1000])
    ax.set_zlabel('Z')

    ax.set_title(name)

    # Provide starting angle for the view.
    ax.view_init(45, -135)

    #run the animation script
    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(zeroframe, data, scatters, displaytype),
                                        interval=10, blit=False, repeat=True)

    #save mp4 file if needed
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(name, dpi=300, writer=writer)

    plt.show()
    
if __name__ == "__main__": 
    
    #define paths of the input files
    pathpp = r"C:\Users\wirth\polybox\MSc\Master Thesis\03_Relevant Files\03_Software\PostprocessedFiles\S01_19112020_T03\S01_19112020_T03_Postprocessed_corr4.pkl"
    pathip = r"C:\Users\wirth\polybox\MSc\Master Thesis\03_Relevant Files\03_Software\PostprocessedFiles\S01_19112020_T03\S01_19112020_T03_Interpreted_corr5.pkl"
      
    #define name of the mp4 output
    name = 'test.mp4'
    
    #choose chain to display
    #body
    #myosuit
    displaytype = 'body'
    
    #define the first frame of the video
    zeroframe = 0
    
    #set the length of the video
    #'Full': animate from the defined first frame to the end of the data
    #[number]: number of animated frames
    length = 1000
    
    
    #load data
    [seg,markercloud,labels,refvecs] = extractdata(pathpp,pathip)
    
    #change representation of data
    data = buildanimdata(seg,markercloud,labels,displaytype)
    
    #run animation
    anim(data,zeroframe,length,name,displaytype,save=True)
    
    