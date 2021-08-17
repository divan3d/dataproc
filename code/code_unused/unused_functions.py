# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:56:28 2021

@author: ksdiv

contains unused finished functions 

"""

import pickle
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import pandas as pd
import statistics


def dyn_places_to_cut(d1):
    """
    finds maximum of shank angle -- corresponds to heel strike

    Parameters
    ----------
    d1 : dataframe containing dynamic trial

    Returns
    -------
    list_of_max : list, contains indices of max values, 0 and last value added 
    to help cut the data later

    """
    max2 = scipy.signal.argrelextrema(d1["mc_shank_angle"].values, np.greater_equal, order = 100)
    size_data = len(d1)
    list_of_max =  max2[0]
    list_of_max = np.append(list_of_max, size_data)
    list_of_max = np.append(0, list_of_max)
    return list_of_max


def dyn_cut_in_ind_gait(d1, list_of_max):
    """
    cuts dynamic trial into individual gait cycles 

    Parameters
    ----------
    d1 : dataframe : containing dynamic trial
    list_of_max : list : of indices of max shank values 

    Returns
    -------
    dict_of_df : dict : containing dataframe of each ind gait cycle

    """ 
    dict_of_df = {}
    for cidx in range(len(list_of_max)-1):
        dict_of_df[cidx] = d1.iloc[list_of_max[cidx]:list_of_max[cidx+1]]  
        
    for key in dict_of_df:
        dict_of_df[key] = dict_of_df[key].reset_index()
    
    return  dict_of_df


def dyn_cut_to_ind_gait_cycle(data_in):
    """
    separates dynamic dataframe into dict of individual gait cycle

    Parameters
    ----------
    data_in : dataframe : dynamic experiment 

    Returns
    -------
    dict_ind_gait : dict : dict of dataframe of separated gait cycle

    """
    l_max = dyn_places_to_cut(data_in)
    dict_ind_gait = dyn_cut_in_ind_gait(data_in, l_max)
    print(" dictionary of individual gait cycles created")    
    return dict_ind_gait

def get_exact_ind(array_of_max, array_peaks):
    """
    separates walking trial by looking at indices of the peaks of walking trial 
    and derivatives of these indices 

    Parameters
    ----------
    array_of_max : np array : output find_peaks, of derivative of indices of 
                   peaks of walking trial
    array_peaks : np array : output of find_peaks, gives indices of peaks of 
                  shank angles during walking trial

    Returns
    -------
    exact_ind : list : contains idx of last peak of previous walking trial and 
                first idx of next walking trial (will not find beginning of 
                very first walking trial or end of last walking trial)

    """
    exact_ind = []
    for val in array_of_max:
        temp = array_peaks[val - 3 : val + 3]
        sub = temp[1:] - temp[:-1]
        amax = np.argmax(sub)
        exact_ind.append(temp[amax])
        exact_ind.append(temp[amax + 1])
    return exact_ind
        

# normalement on aurait les débuts de la marche dans les positions paires (0,2,..)
# et les fins de marche dans les positions impaires (1,3,..)

def get_earlier_and_later_idx(places_to_cut, offset):
    """
    removes or adds offset depending on wether its beginning of end of walking

    Parameters
    ----------
    places_to_cut : list : of indices that are the beginning or end of walking
                    trial, even - beginning, odd - end 
    offset : int : offset to remove/add to indices

    Returns
    -------
    updated_ptc : list : of indices with offset to separate whole trial into 
                  only the walking trials 

    """
    updated_ptc = [0] * len(places_to_cut)
    # retire offset au début (paire):
    beg_idx = range(0,len(places_to_cut),2)
    for bidx in beg_idx:
        updated_ptc[bidx] = places_to_cut[bidx] - offset
    end_idx = range(1,len(places_to_cut),2)
    for eidx in end_idx:
        updated_ptc[eidx] = places_to_cut[eidx] + offset
    return updated_ptc


def find_indice_to_cut(data_in):
    """
    returns indices to separate whole trial to get only the parts where subject
    is walking, does this by looking at shank angles

    Parameters
    ----------
    data_in : pd dataframe : whole trial 

    Returns
    -------
    places_to_cut : list : 
        even values : beginning of walking, 
        odd values : end of walking
                
    """
    # get indices of places where angle > 5
    peaks_shank = scipy.signal.find_peaks(data_in["no_mc_shank_angle"], height=(5))
    p_peaks_shank = peaks_shank[0]
    # get gradient of indices 
    grad_peaks = np.gradient(p_peaks_shank)
    # find indices of "steps"/ sudden increase in indices of shank angles > 5
    temp_peaks = scipy.signal.find_peaks(grad_peaks, height = 500)
    bplaces_to_cut = get_exact_ind(temp_peaks[0], p_peaks_shank)
    # add first value - beginning of first walking trial
    places_to_cut = get_earlier_and_later_idx(bplaces_to_cut, 300)
    bplaces_to_cut.insert(0,p_peaks_shank[0])
    
    return places_to_cut