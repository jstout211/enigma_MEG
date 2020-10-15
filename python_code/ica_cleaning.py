#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:38:50 2020

@author: stoutjd
"""


# from hv_proc import test_config
# filename = test_config.rest['meg']
# raw = mne.io.read_raw_ctf(filename, preload=True)
# raw.apply_gradient_compensation(3)
# raw.resample(300)
# raw.filter(1.0, None)
# raw.notch_filter([60,120])



import os, glob
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

def get_cardiac_epochs(filename):
    #tmp = ica.get_sources(raw)
    #cardiac = tmp[ecg_idx, :]
    import numpy as np
    import pylab as plt
    
    event_id = 999
    raw = mne.io.read_raw_ctf(filename, preload=True)
    #raw.filter(1.0, None)
    ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, event_id, ch_name='MLF23-1609')
    tmin, tmax = -0.1, 0.1 
    raw.del_proj()
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=False, include=['MLF23-1609'], exclude='bads') 
    epochs = mne.Epochs(raw, ecg_events, event_id, tmin, tmax, picks=picks)
    data = epochs.get_data()  
    plt.plot(1e3 * epochs.times, np.squeeze(data).T) 
    plt.xlabel('Times (ms)')
    plt.ylabel('ECG')
    plt.show()    


def get_datasets():
    top_dir = '/data/MEG/rest_ica'
    import glob 
    files = glob.glob(os.path.join(top_dir, '*_rest*.ds'))
    return files
    
def review_data():
    import subprocess
    top_dir = '/data/MEG/rest_ica'
    import glob 
    files = glob.glob(os.path.join(top_dir, '*_rest*.ds'))
    os.chdir(top_dir)
    files = [os.path.basename(i) for i in files]
    idx=0
    answer = 'y'
    while answer.lower() != 'n':
        print(files[idx])
        cmd = 'DataEditor -data {} -f '.format(files[idx])
        subprocess.run(cmd, shell=True)
        idx+=1
        answer=input('Continue y/n:')
    print('Done')
    

# def calc_ica(filename):
#     raw = mne.io.read_raw_ctf(filename, preload=True)
#     raw.apply_gradient_compensation(3)
#     raw.resample(300)
#     raw.filter(1.0, None)
#     raw.notch_filter([60,120])
    
#     rand_state = range(1,10)
#     for rstate in rand_state:
#         ica = ICA(n_components=30, random_state=rstate)
#         ica.fit(raw)
        
#         #raw.load_data()
#         #ica.plot_sources(raw)
        
#         out_filename = os.path.join(filename, 'rest_{}-ica.fif'.format(str(rstate)))
#         ica.save(out_filename)
        
def calc_ica(filename, outfilename):
    raw = mne.io.read_raw_ctf(filename, preload=True)
    raw.apply_gradient_compensation(3)
    raw.resample(300)
    raw.filter(1.0, None)
    raw.notch_filter([60,120])
    
    ica = ICA(n_components=30, random_state=0)
    ica.fit(raw)
    
    #raw.load_data()
    #ica.plot_sources(raw)
    
    
    
    out_filename = outfilename+'-ica.fif'
    ica.save(out_filename)        



        
if __name__=='__main__':
    import sys
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    calc_ica(filename, outfilename)
    #get_cardiac_epochs(filename)
    