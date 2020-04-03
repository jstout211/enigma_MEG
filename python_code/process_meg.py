#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:14:08 2020

@author: stoutjd
"""



### Import Section
import numpy as np
import matplotlib.pyplot as plt
import os, copy, glob 
import mne
from mne.minimum_norm import apply_inverse_raw
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import pandas as pd
from scipy.signal import hilbert, butter, lfilter, filtfilt
import logging, inspect

try:
    from mne.connectivity import envelope
except:
    pass
    
def setup_logging(logfile):
    logging.basicConfig(filename=logfile, level=logging.DEBUG, filemode='a',
                        format='%(asctime)s %(message)s')

'''
########################### Structure of data analysis########################
Functions are called from the project import and wrapped with subject loops
Helper functions include:
env_corr_freq_loop(freq_bands=None, sfreq=None, label_ts_orig=None, outfolder=None, SUBJID=None)
    input is label timeseries and loops over all frequency pairs to output connectivity matrices
write_matrices(outfolder=None, meas=None, fmin=None, fmax=None, con_res=None)
    writes numpy matrix and image of connectivity matrix
apply_epoch_hilbert(stcs=[], sfreq=None, fmin=None, fmax=None)
apply_epoch_label_ts_hilbert(label_ts=[], sfreq=None, fmin=None, fmax=None)
apply_epoch_filter(label_ts=[], sfreq=None, fmin=None, fmax=None)
'''

def apply_epoch_hilbert(stcs=[], sfreq=None, fmin=None, fmax=None):
    '''Feed in epochs of source data to generate the hilbert transform of the 
    source data.  This will add edge padding (50% per side) and filter, then 
    hilbert transform, then remove the padding
    INPUTS:
    stcs - epoched source projection
    sfreq - sampling frequency
    fmin - low frequency
    fmax - high frequency (i.e. low pass frequency)
    
    Note the padding size is not adjustable to the frequency << May need to fix or optimize
    '''
    for idx in range(len(stcs)):
        mat=stcs[idx].data    
        time_val=mat.shape[1]
        time_pad1=range(-time_val/2,0)
        time_pad2=range(0, time_val/2)
        mat_pad=np.concatenate([mat[:, time_pad1], mat, mat[:, time_pad2]], axis=1)
        nyq = sfreq * 0.5
        b,a = butter(4, [fmin/nyq, fmax/nyq], btype='band')
        hilb_pad=np.abs(hilbert(lfilter(b,a,mat_pad)))
        stcs[idx]._data=hilb_pad[:,len(time_pad1): mat.shape[1]+len(time_pad1)]
    return stcs

def apply_epoch_label_ts_hilbert(label_ts=[], sfreq=None, fmin=None, fmax=None):
    '''Feed in epochs of source data to generate the hilbert transform of the 
    source data.  This will add edge padding (50% per side) and filter, then 
    hilbert transform, then remove the padding
    INPUTS:
    stcs - epoched source projection
    sfreq - sampling frequency
    fmin - low frequency
    fmax - high frequency (i.e. low pass frequency)
    
    Note the padding size is not adjustable to the frequency << May need to fix or optimize
    '''
    for idx in range(len(label_ts)):
        mat=label_ts[idx] #.data    
        time_val=label_ts[idx].shape[1] #.data.shape[1] #mat.shape[1]
        time_pad1=range(-time_val/2,0)
        time_pad2=range(0, time_val/2)
        ## Pad from the middle out to reduce discontinuity jumps
        mat_pad=np.concatenate([mat[:, time_pad1], mat, mat[:, time_pad2[-1:0:-1]]], axis=1) 
        nyq = sfreq * 0.5
        b,a = butter(4, [fmin/nyq, fmax/nyq], btype='band')
        hilb_pad=np.abs(hilbert(filtfilt(b,a,mat_pad)))
        label_ts[idx]=hilb_pad[:,len(time_pad1): mat.shape[1]+len(time_pad1)]
    return label_ts

def apply_epoch_filter(label_ts=[], sfreq=None, fmin=None, fmax=None):
    '''Feed in epochs add edge padding (50% per side) and filter (forward and backward), then 
    then remove the padding
    INPUTS:
    label_ts - parcellated time series
    sfreq - sampling frequency
    fmin - low frequency
    fmax - high frequency (i.e. low pass frequency)
    
    Note the padding size is not adjustable to the frequency << May need to fix or optimize
    '''
    for idx in range(len(label_ts)):
        mat=label_ts[idx] #.data    
        time_val=label_ts[idx].shape[1] #.data.shape[1] #mat.shape[1]
        time_pad1=range(-time_val/2,0)
        time_pad2=range(0, time_val/2)
        ## Pad from the middle out to reduce discontinuity jumps
        mat_pad=np.concatenate([mat[:, time_pad1], mat, mat[:, time_pad2[-1:0:-1]]], axis=1) 
        nyq = sfreq * 0.5
        b,a = butter(4, [fmin/nyq, fmax/nyq], btype='band')
        filt_pad=filtfilt(b,a,mat_pad)
        label_ts[idx]=filt_pad[:,len(time_pad1): mat.shape[1]+len(time_pad1)]
    return label_ts

 
############################ END HELPER FUNCTIONS ############################   
##############################################################################


def compute_rest_phase(SUBJID,DATATYPE='RestEO', RUN=None, PROJECT=None, COLUMN='PROC_CLEAN', outfolder=None):
    '''Calculate the '''
    er_raw=mne.io.read_raw_fif(subjpd.loc['emptyroom_1']['FilePath_tSSS'], preload=True)
    cov = mne.compute_raw_covariance(er_raw)

    #Configure Logging
    setup_logging(PROJECT.logfile)
    logging.info('Started:'+inspect.currentframe().f_code.co_name+ ':'+SUBJID)
    
    #Setup output
    if  outfolder==None:
        raise ValueError('No outfolder')
    outfolder=os.path.join(outfolder,SUBJID)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)    
    
    for RUN in ['1','2','3']:
        data_index=DATATYPE+'_'+str(RUN)
        trans_fname=subjpd.loc[data_index, 'Anat_Coreg']
        src_fname=subjpd.loc[data_index, 'Anat_SRC']
        bem_sol_fname=subjpd.loc[data_index, 'Anat_BEM_SOL']
        er_fname=subjpd.loc['emptyroom_1', 'PROC_INIT']
        fwd_fname=subjpd.loc['RestEO_'+str(RUN),'Anat_FWD_SOL']
        freq_bands=PROJECT.params['data'][DATATYPE]['freq_bands']
        SUBJECTS_DIR=os.path.join(PROJECT.project_dir,SUBJID)
        fs_subj='Anatomy'
        
        fwd = mne.read_forward_solution(fwd_fname)
        raw_fname=subjpd.ix[data_index, COLUMN]
        raw = mne.io.Raw(raw_fname, preload=True)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, exclude='bads')
        raw.pick_types(meg=True, eeg=False, stim=False, exclude='bads')

        ##Create the inverse operator 
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, 
                                                                  loose=0.2)
        
        events = mne.make_fixed_length_events(raw, 1, start = 0, stop = None, 
                                              duration = 4.0, first_samp=True)
        
        epochs = mne.Epochs(raw, events , event_id = 1, tmin=-2.0, tmax = 2.0,
                        baseline=(-2.0,2.0), preload=True)
        # Compute inverse solution and for each epoch. By using "return_generator=True"
        # stcs will be a generator object instead of a list.
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
    label_ts_save(label_ts=label_ts, outfolder=outfolder)
    logging.info('Completed:'+inspect.currentframe().f_code.co_name+ ':label_ts:'+SUBJID)
    phase_freq_loop(freq_bands=freq_bands, sfreq=raw.info['sfreq'], label_ts_orig=label_ts, outfolder=outfolder, SUBJID=SUBJID)                    
    logging.info('Completed:'+inspect.currentframe().f_code.co_name+ ':rest_phase_conn:'+SUBJID)


#def get_448_labels(PROJECT=None, subjid=None, subjects_dir=None):
#    '''Link the FSaverage to each subject folder, then morph the 448 atlas from 
#    fsaverage to the subject surface for processing.
#    If fsaverage and single subject morphing are done, just return the labels'''
#    os.path.join(project_dir,subjid,'morph-maps/Anatomy-fsaverage-morph.fif')
#
     
        
def compute_rest_envelope(SUBJID,DATATYPE='RestEO', RUN=None, PROJECT=None, COLUMN='PROC_CLEAN', outfolder=None):
    subjpd=PROJECT.load_subj_pd(SUBJID)
    
    #Configure Logging
    setup_logging(PROJECT.logfile)
    logging.info('Started:'+inspect.currentframe().f_code.co_name+ ':'+SUBJID)
    
    #Setup output
    if  outfolder==None:
        raise ValueError('No outfolder')
    outfolder=os.path.join(outfolder,SUBJID)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)    
    
    #Estimate Noise Covariance    
    er_raw=mne.io.read_raw_fif(subjpd.loc['emptyroom_1']['FilePath_tSSS'], preload=True)
    cov = mne.compute_raw_covariance(er_raw)

    for RUN in ['1','2','3']:
        data_index=DATATYPE+'_'+str(RUN)
        trans_fname=subjpd.loc[data_index, 'Anat_Coreg']
        src_fname=subjpd.loc[data_index, 'Anat_SRC']
        bem_sol_fname=subjpd.loc[data_index, 'Anat_BEM_SOL']
        er_fname=subjpd.loc['emptyroom_1', 'PROC_INIT']
        fwd_fname=subjpd.loc[DATATYPE+'_'+str(RUN),'Anat_FWD_SOL']
        freq_bands=PROJECT.params['data'][DATATYPE]['freq_bands']
        SUBJECTS_DIR=os.path.join(PROJECT.project_dir,SUBJID)
        fs_subj='Anatomy'
        
        fwd = mne.read_forward_solution(fwd_fname)
        raw_fname=subjpd.loc[data_index, COLUMN]
        raw = mne.io.Raw(raw_fname, preload=True)
        raw.resample(200, n_jobs=10)
                
        picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, exclude='bads')
        raw.pick_types(meg=True, eeg=False, stim=False, exclude='bads')
        ##Create the inverse operator 
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, 
                                                                  loose=0.2)
        
        events = mne.make_fixed_length_events(raw, 1, start = 0, stop = None, 
                                              duration = 4.0, first_samp=True)
        
        epochs = mne.Epochs(raw, events , event_id = 1, tmin=-2.0, tmax = 2.0,
                        baseline=(-2.0,2.0), preload=True)
        
        # Compute inverse solution and for each epoch. By using "return_generator=True"
        # stcs will be a generator object instead of a list.
        snr = 1.0  # use lower SNR for single epochs
        lambda2 = 1.0 / snr ** 2
        method = 'MNE' #"dSPM"  # use dSPM method (could also be MNE or sLORETA)
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                    pick_ori="normal", return_generator=False)
        src = inverse_operator['src']  
        # Extract Label timeseries
        labels_lh=mne.read_labels_from_annot(fs_subj, parc='aparc_sub',
                                            subjects_dir=SUBJECTS_DIR, hemi='lh') 
        labels_rh=mne.read_labels_from_annot(fs_subj, parc='aparc_sub',
                                            subjects_dir=SUBJECTS_DIR, hemi='rh') 
        labels=labels_lh + labels_rh 
        temp_label_ts=mne.extract_label_time_course(stcs, labels, src, mode='mean_flip') #, return_generator=False)
        #Concatenate the label timeseries over runs        
        if 'label_ts' in vars():
            label_ts+=temp_label_ts
        else:
            label_ts=temp_label_ts
    label_ts_orig=copy.copy(label_ts)
    label_ts_save(label_ts=label_ts_orig, outfolder=outfolder)
    logging.info('Completed:'+inspect.currentframe().f_code.co_name+ ':label_ts:'+SUBJID)
    env_corr_freq_loop(freq_bands=freq_bands, sfreq=raw.info['sfreq'], label_ts_orig=label_ts_orig, outfolder=outfolder, SUBJID=SUBJID)                    
    logging.info('Completed:'+inspect.currentframe().f_code.co_name+ ':rest_conn:'+SUBJID)
 
##EXAMPLE SUBJECT  - Comment out after use
#from project import MEGPR
#SUBJID='EC1002'
#DATATYPE='SM'
#RUN='1'
#PROJECT=MEGPR.database
#COLUMN='PROC_CLEAN'
#outfolder='/data/MEG/CONN_SM_448_folders/ECP_448_SM_parcel_mean_flip'
#            

           
def compute_SM_envelope(SUBJID,DATATYPE='SM', RUN=None, PROJECT=None, COLUMN='PROC_CLEAN', outfolder=None):
    '''Command produces connectivity evaluation during the semantic portion
    of the story/math task'''
    subjpd=PROJECT.load_subj_pd(SUBJID)
    
    #Configure Logging
    setup_logging(PROJECT.logfile)
    logging.info('Started:'+inspect.currentframe().f_code.co_name+ ':'+SUBJID)
    
    #Setup output
    if  outfolder==None:
        raise ValueError('No outfolder')
    outfolder=os.path.join(outfolder,SUBJID)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)    
    
    #Estimate Noise Covariance    
    er_raw=mne.io.read_raw_fif(subjpd.loc['emptyroom_1']['FilePath_tSSS'], preload=True)
    cov = mne.compute_raw_covariance(er_raw)

    for RUN in ['1','2']:
        data_index=DATATYPE+'_'+str(RUN)
        trans_fname=subjpd.loc[data_index, 'Anat_Coreg']
        src_fname=subjpd.loc[data_index, 'Anat_SRC']
        bem_sol_fname=subjpd.loc[data_index, 'Anat_BEM_SOL']
        er_fname=subjpd.loc['emptyroom_1', 'PROC_INIT']
        fwd_fname=subjpd.loc[DATATYPE+'_'+str(RUN),'Anat_FWD_SOL']
        freq_bands=PROJECT.params['data'][DATATYPE]['freq_bands']
        SUBJECTS_DIR=os.path.join(PROJECT.project_dir,SUBJID)
        fs_subj='Anatomy'
        
        fwd = mne.read_forward_solution(fwd_fname)
        raw_fname=subjpd.loc[data_index, COLUMN]
        raw = mne.io.Raw(raw_fname, preload=True)
        raw.resample(200, n_jobs=10)
                
        raw.pick_types(meg=True, eeg=False, stim=True, exclude='bads')
        ##Create the inverse operator 
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, 
                                                                  loose=0.2)
        
        ################### Process Triggers #####################
        #Break up 16s block data into 4x4 second trials
        trig_idx=raw.info['ch_names'].index('STI101')
        trig=raw.copy().pick_channels(['STI101'])
        trig._data[trig._data!=56]=0 #Eliminate all triggers except the story
        
        indices=np.where(trig._data==56)
        timestep=4 #in seconds
        set1=indices[1]+raw.info['sfreq']*timestep*1  #4second advance
        set2=indices[1]+raw.info['sfreq']*timestep*2 #8seconds
        set3=indices[1]+raw.info['sfreq']*timestep*3 #12seconds
        trig._data[0,np.concatenate([set1,set2,set3]).astype(int)]=56
        raw._data[trig_idx,:]=trig._data          

        ######################################################################
        ## Process Events
        event_id = {'Story_onset': 56} 
        tmin = 0.0
        tmax = 4.0
        events=mne.find_events(raw, stim_channel='STI101')
        baseline = (None, 4.0)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=0.0,
                    tmax=4.0, baseline=baseline, preload=True) #,picks=('mag','grad')) #, 'eog'))  
        epochs.pick_types(meg=True) 
        ## Calculate Inverse
        snr = 1.0  # use lower SNR for single epochs
        lambda2 = 1.0 / snr ** 2
        method = 'MNE' #"dSPM"  # use dSPM method (could also be MNE or sLORETA)
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                    pick_ori="normal", return_generator=False)
        src = inverse_operator['src']  
        # Extract Label timeseries
        labels_lh=mne.read_labels_from_annot(fs_subj, parc='aparc_sub',
                                            subjects_dir=SUBJECTS_DIR, hemi='lh') 
        labels_rh=mne.read_labels_from_annot(fs_subj, parc='aparc_sub',
                                            subjects_dir=SUBJECTS_DIR, hemi='rh') 
        labels=labels_lh + labels_rh 
        temp_label_ts=mne.extract_label_time_course(stcs, labels, src, mode='mean_flip') #, return_generator=False)
        #Concatenate the label timeseries over runs        
        if 'label_ts' in vars():
            label_ts+=temp_label_ts
        else:
            label_ts=temp_label_ts
    label_ts_orig=copy.copy(label_ts)
    label_ts_save(label_ts=label_ts_orig, outfolder=outfolder)
    logging.info('Completed:'+inspect.currentframe().f_code.co_name+ ':label_ts:'+SUBJID)
    env_corr_freq_loop(freq_bands=freq_bands, sfreq=raw.info['sfreq'], label_ts_orig=label_ts_orig, outfolder=outfolder, SUBJID=SUBJID)                    
    logging.info('Completed:'+inspect.currentframe().f_code.co_name+ ':sm_conn:'+SUBJID)
            
