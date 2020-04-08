#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:08:12 2020

@author: stoutjd
"""
import os
import mne

#@save_numpy_output
def load_test_data():
    from hv_proc import test_config
    filename=test_config.rest['meg']
    raw=load_data(filename)
    return raw

def save_numpy_output(func):
    def wrapper(*args,**kwargs):
        #print("Something is happening before the function is called.")
        output=func(*args,**kwargs)
        output.save(func.__name__+'.npy')
        print("Saved data to {}".format(func.__name__+'.npy'))
    return wrapper

################

def check_datatype(filename):
    '''Check datatype based on the vendor naming convention'''
    if os.path.splitext(filename)[-1] == '.ds':
        return 'ctf'
    elif os.path.splitext(filename)[-1] == '.fif':
        return 'elekta'
    elif os.path.splitext(filename)[-1] == '.4d':
        return '4d'
    elif os.path.splitext(filename)[-1] == '.sqd':
        return 'kit'
    else:
        raise ValueError('Could not detect datatype')
        
def return_dataloader(datatype):
    '''Return the dataset loader for this dataset'''
    if datatype == 'ctf':
        return mne.io.read_raw_ctf
    if datatype == 'elekta':
        return mne.io.read_raw_fif
    if datatype == '4d':
        return mne.io.read_raw_bti
    if datatype == 'kit':
        return mne.io.read_raw_kit

def load_data(filename):
    datatype = check_datatype(filename)
    dataloader = return_dataloader(datatype)
    raw = dataloader(filename, preload=True)
    return raw

def calculate_inverse(epochs, outfolder=None):
    cov = mne.compute_covariance(epochs)
    cov.save(os.path.join(outfolder, 'rest-cov.fif'))
    
    
    
    
    
    
    # trans_fname=subjpd.loc[data_index, 'Anat_Coreg']
    # src_fname=subjpd.loc[data_index, 'Anat_SRC']
    # bem_sol_fname=subjpd.loc[data_index, 'Anat_BEM_SOL']
    # er_fname=subjpd.loc['emptyroom_1', 'PROC_INIT']
    # fwd_fname=subjpd.loc['RestEO_'+str(RUN),'Anat_FWD_SOL']
    # freq_bands=PROJECT.params['data'][DATATYPE]['freq_bands']
    # SUBJECTS_DIR=os.path.join(PROJECT.project_dir,SUBJID)
    # fs_subj='Anatomy'
    
    # fwd = mne.read_forward_solution(fwd_fname)
    # raw_fname=subjpd.ix[data_index, COLUMN]
    # raw = mne.io.Raw(raw_fname, preload=True)
    # picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, exclude='bads')
    # raw.pick_types(meg=True, eeg=False, stim=False, exclude='bads')

    # ##Create the inverse operator 
    # inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, 
    #                                                           loose=0.2)
    
    
    
    



def main(filename):
    raw=load_data(filename)
    
    raw.resample(300)
    raw.filter(0.3, None)
    raw.notch_filter([60,120])
    
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)
    
    
    #Clear memory
    del raw
    
    #Drop bad epochs and channels
    ##
    
    #Calculate covariance
    cov = mne.compute_covariance(epochs)
    
    #Filter data
        
    
    #Calculate Inverse solution
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = 'MNE' #"dSPM"  # use dSPM method (could also be MNE or sLORETA)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                pick_ori="normal", return_generator=False)
    src = inverse_operator['src']  
    labels_lh=mne.read_labels_from_annot(fs_subj, parc='aparc_sub',
                                        subjects_dir=SUBJECTS_DIR, hemi='lh') 
    labels_rh=mne.read_labels_from_annot(fs_subj, parc='aparc_sub',
                                        subjects_dir=SUBJECTS_DIR, hemi='rh') 
    labels=labels_lh + labels_rh 
    temp_label_ts=mne.extract_label_time_course(stcs, labels, src, mode='mean_flip') 
    #Concatenate the label timeseries        
    if 'label_ts' in vars():
        label_ts+=temp_label_ts
    else:
        label_ts=temp_label_ts

    

    


 # Check data type and load data
 # Downsample to 200Hz
 # Split to 1 second epochs
 # Reject sensor level data at a specific threshold
 # Calculate broad band dSPM inverse solution
 # Filter the data into bands (1-3, 3-6, 8-12, 13-35, 35-55)
 # Project the data to parcels and create parcel time series
 # Calculate relative power in each band and parcel    
    
   
    
if __name__=='__main__':
    import sys
    load_data(sys.argv[1])


