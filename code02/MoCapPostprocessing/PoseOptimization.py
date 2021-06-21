# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 07:00:53 2020

@author: Marc
"""
#import libraries and packages
import numpy as np
import math as m
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import datetime


#function to calculate resulting force and moment on CoM of point cloud
def actingforces(ref,meas,weights=[-1]):
    if weights[0] == -1:
        weights=np.ones(len(ref))
    
    #calculate Center of Moment
    com = ref.sum(axis=0)/len(ref)
    
    #set CoM to origin
    refpos_corr = ref - com
    
    #difference of measured posture and estimation
    dpos = meas - ref
    
    #initialization of Force and Moment vector
    F = np.zeros_like(dpos)
    M = np.zeros_like(dpos)
    
    #calculation of resulting force
    for i in range(0,len(ref)):
        F[i,:] = dpos[i,:]*weights[i]            
    Ftot = F.sum(axis=0)/len(F)
    
    #calculation of resulting moment
    for i in range(0,len(ref)):
        M[i,:] = np.cross(refpos_corr[i,:],F[i,:])        
    Mtot = M.sum(axis=0)/len(M)
    
    return Ftot,Mtot

#function to calculate adapted point cloud
def newpointset(ref,meas,weights=[-1],scalingF=10,scalingM=50):
    
    if weights[0] == -1:
        weights=np.ones(len(ref))
    
    #get the forces acting on the cloud
    out = actingforces(ref,meas,weights)
    Ftot = out[0]
    Mtot = out[1]
    
    #calculate CoM and corrected position (CoM in origin)
    com = ref.sum(axis=0)/len(ref)
    refpos_corr = ref - com  
    measpos_corr = meas - com
    
    #move points along the acting force
    points = ref + Ftot/(scalingF*np.amax(weights))
    
    angles = np.zeros_like(weights)
    for i in range(0,len(angles)):
        angles[i]=np.dot(np.squeeze(np.asarray(refpos_corr[i,:])),np.squeeze(np.asarray(measpos_corr[i,:])))
        angles[i]=angles[i]/(np.linalg.norm(refpos_corr[i,:])*np.linalg.norm(measpos_corr[i,:]))
        if np.abs(angles[i])>1.:
            angles[i]=angles[i]/np.abs(angles[i])
        angles[i]=np.arccos(angles[i])
        
    #rotate points along resulting moment
    for i in range(0,len(points)):
        angle = np.amax(angles)/scalingM
        
        Mnorm = Mtot / np.linalg.norm(Mtot)
        
        comp_along_M = np.dot(points[i,:],Mnorm)*Mnorm
        point_flat =  points[i,:]-comp_along_M
        
        points[i,:]=point_flat*m.cos(angle)+np.cross(Mnorm,point_flat/np.linalg.norm(point_flat))*np.linalg.norm(point_flat)*m.sin(angle)+comp_along_M
          
    
    return points

#custom function to align a measured marker set to a custom marker set
def optimize(refpos,measpos,weights=[-1],graph=False,printout=False,n=-1):
    
    #start time measurement
    starttime = datetime.datetime.now()
    
    #initialize forces and error for initial data sets
    forces = actingforces(refpos,measpos,weights)
    err = np.linalg.norm(forces[0])+np.linalg.norm(forces[1])
    forceold = np.linalg.norm(forces[0])
    momold   = np.linalg.norm(forces[1])

    #initialize new position matrix, counter and initial scalings
    newpos = refpos
    i = 0
    Fscale = 10
    Mscale = 50
 
    #loop until error has reached threshold
    while err > 1:
        newpos = newpointset(newpos,measpos,weights,scalingF=Fscale,scalingM=Mscale)
        forces = actingforces(newpos,measpos,weights)
        errnew = np.linalg.norm(forces[0])+np.linalg.norm(forces[1])
    
        #detect swinging and adapt scaling
        if errnew > err:
            if forceold < np.linalg.norm(forces[0]):
                Fscale +=1
                if momold < np.linalg.norm(forces[1]):
                    Mscale +=1

        #save old values
        forceold = np.linalg.norm(forces[0])
        momold   = np.linalg.norm(forces[1])
        err = errnew    
  
        i += 1      
        if i>1000:
            break
    dur=datetime.datetime.now()-starttime
    if printout==True:
        print("Execution time:",dur)
        print("Error:",err)

    
    if graph==True:    
        #plot estimated position and measured point cloud  
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.scatter(measpos[:,0], measpos[:,1], measpos[:,2],c='red')
        ax.scatter(newpos[:,0], newpos[:,1], newpos[:,2],c='blue')
        ax.scatter(refpos[:,0], refpos[:,1], refpos[:,2],c='green')
        
    ret = [newpos,n]
    
    if n==-1:
        ret = newpos
    
    return ret

#function to optimize segment alignement based on a custom approach
#input: reference marker configuration (ref), measured marker data of a segment (seg), weights (do nothing, but needed for syntax)
def optimsegment(ref,seg,weights=[-1]):
    #start time measurement
    prevtime = datetime.datetime.now()
    
    #set up empty list of corrected segment data
    corrseg = [0]*len(seg)
    
    #set frame counter to zero
    frameno = 0
    
    #iterate over all frames
    for i in range(0,len(seg)):
        #return corrected segment using a custom approach
        corrseg[i] = optimize(ref[:,0:3],seg[i][:,0:3],weights)
        frameno +=1
        
        #print progress to console
        if frameno % 100 == 0:
            progtime = datetime.datetime.now() - prevtime
            print('Progress: Frame',frameno,'(Time effort:',progtime,')')
            prevtime += progtime
                            
    return corrseg

#function to align a reference markerset (x_ref) to a measured marker set (x_i)
def optimize_Soederkvist(x_ref,x_i):
    
    try:
        #bring to correct form
        x_ref = x_ref.T
        x_i = x_i.T
        
        #calculate center of masses
        xiq = np.average(x_i,axis=1)
        xq  = np.average(x_ref,axis=1)
        
        #set up empty matrices
        A = np.zeros_like(x_ref,dtype=float)
        B = np.zeros_like(x_i,dtype=float)
        
        #correct marker sets such that they are centered around the origin
        for i in range(0,len(x_ref.T)):
            A[:,i] = x_ref[:,i] - xq
            B[:,i] = x_i[:,i] - xiq
            
        #singular value decomposition
        u, sigma, v = np.linalg.svd(np.matmul(A,B.T), full_matrices=True)
        v = v.T
        sigma = np.diag(sigma)
        
        detvu = np.linalg.det(np.matmul(v,u.T))
        
        diagarray = (len(v)-1)*[1]
        diagarray.append(detvu)
        diagarray = np.asarray(diagarray)
        diagmat = np.diag(diagarray)
        
        Q = np.matmul(diagmat,u.T)
        Q = np.matmul(v,Q)
        
        t2 = xiq - np.matmul(Q,xq)
        
        scale = np.trace(np.matmul(Q.T,np.matmul(B,A.T)))/np.trace(np.matmul(A.T,A))
        
        residual_matrix = x_i - np.matmul(Q,x_ref)
        for i in range(0,len(x_i.T)):
            residual_matrix[:,i] = residual_matrix[:,i] - t2
            
        residual = np.trace(np.matmul(residual_matrix.T,residual_matrix))
        
        res_norm = (residual / len(x_i.T))**0.5
        
        ret = scale * np.matmul(Q,x_ref)
        for i in range(0,len(x_i.T)):
            ret[:,i] = ret[:,i] + t2
          
        #put into correct return position
        ret = ret.T
        
        u_A, sigma_A, v_A = np.linalg.svd(A, full_matrices=True)
        u_A = u_A.T
        princ_dir = []
        princ_dir.append(u_A[:,0])
        princ_dir.append(u_A[:,1])
        princ_dir.append(u_A[:,2])
        
    except:
        ret = np.zeros_like(x_ref)
        res_norm = 0
        princ_dir = [0,0,0]
    return ret,res_norm,princ_dir


#function to optimize segment alignement based on Soederkvists approach
#input: reference marker configuration (ref), measured marker data of a segment (seg), weights (do nothing, but needed for syntax)
def optimsegment_Soederkvist(ref,seg,weights=[-1]):
    #start time measurement
    prevtime = datetime.datetime.now()
    
    #initialize empty vectors
    corrseg = [0]*len(seg)
    res_norm = [0]*len(seg)
    princ_dir = [0]*len(seg)
    
    #set frame number to zero
    frameno = 0
    
    #iterate over all frames
    for i in range(0,len(seg)):
        #return the optimized marker positions, the residuals and the principa directions
        corrseg[i],res_norm[i],princ_dir_element = optimize_Soederkvist(ref[:,0:3],seg[i][:,0:3])
#        corrseg[i],res_norm[i],princ_dir_element = optimize_Soederkvist(ref[:,0:3],seg[:,0:3])
        princ_dir[i] = princ_dir_element
        
        #check if calculation successful, if not, repeat with custom algorithm
        if np.all(corrseg[i]) == 0:
            corrseg[i] = optimize(ref[:,0:3],seg[i][:,0:3],weights)
            
        frameno +=1
        
        #output progress to console
        if frameno % 100 == 0:
            progtime = datetime.datetime.now() - prevtime
            print('Progress: Frame',frameno,'(Time effort:',progtime,')')
            prevtime += progtime
                    
        
    return corrseg,res_norm,princ_dir

