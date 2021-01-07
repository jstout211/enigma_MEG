#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:38:50 2020

@author: stoutjd
"""

import os, glob
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

# def get_cardiac_epochs(filename):
#     #tmp = ica.get_sources(raw)
#     #cardiac = tmp[ecg_idx, :]
#     import numpy as np
#     import pylab as plt
    
#     event_id = 999
#     raw = mne.io.read_raw_ctf(filename, preload=True)
#     #raw.filter(1.0, None)
#     ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, event_id, ch_name='MLF23-1609')
#     tmin, tmax = -0.1, 0.1 
#     raw.del_proj()
#     picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=False, include=['MLF23-1609'], exclude='bads') 
#     epochs = mne.Epochs(raw, ecg_events, event_id, tmin, tmax, picks=picks)
#     data = epochs.get_data()  
#     plt.plot(1e3 * epochs.times, np.squeeze(data).T) 
#     plt.xlabel('Times (ms)')
#     plt.ylabel('ECG')
#     plt.show()    


# def get_datasets():
#     top_dir = '/data/MEG/rest_ica'
#     import glob 
#     files = glob.glob(os.path.join(top_dir, '*_rest*.ds'))
#     return files
    
# def review_data():
#     import subprocess
#     top_dir = '/data/MEG/rest_ica'
#     import glob 
#     files = glob.glob(os.path.join(top_dir, '*_rest*.ds'))
#     os.chdir(top_dir)
#     files = [os.path.basename(i) for i in files]
#     idx=0
#     answer = 'y'
#     while answer.lower() != 'n':
#         print(files[idx])
#         cmd = 'DataEditor -data {} -f '.format(files[idx])
#         subprocess.run(cmd, shell=True)
#         idx+=1
#         answer=input('Continue y/n:')
#     print('Done')
    
     
def calc_ica(filename, outbasename=None, mains_freq=60):
    if filename[-2:]=='ds':
        raw = mne.io.read_raw_ctf(filename, preload=True)
        raw.apply_gradient_compensation(3)
        
        #Hack to get around extraneous channels from software upgrade
        ch_keeps = [i for i in raw.ch_names if (i[0]=='M')|(i[0]=='E')]
        raw.pick_channels(ch_keeps)
    if filename[-3:]=='fif':
        raw = mne.io.read_raw_fif(filename, preload=True)
    if filename[-4:]=='rfDC':
        raw = mne.io.read_raw_bti(filename, preload=True, head_shape_fname=None)
     
    resample_freq = 300
    notch_freqs = range(mains_freq, int(resample_freq * 2/3), mains_freq)
    raw.notch_filter(notch_freqs)
    
    raw.resample(resample_freq)
    raw.filter(1.0, None)  #Necessary for stable ICA calcs  (stationarity assumption?)
    
    if outbasename != None:
        file_base = outbasename #Necessary for 4D datasets
    else:
        file_base = os.path.basename(filename)
        file_base = os.path.splitext(file_base)[0]	 
    
    rand_state = range(0,10)
    raw.save(file_base+'_300srate.fif') #Save with EEG
    raw.pick_types(meg=True, eeg=False)
    for rstate in rand_state:
        ica = ICA(n_components=25, random_state=rstate)
        ica.fit(raw)
        out_filename = file_base + '_{}-ica.fif'.format(str(rstate))
        ica.save(out_filename)

        

        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', 
                      help='MEG dataset')
    parser.add_argument('-output_base',
                      help='''Required for 4D datasets.  Will assume filename
                      prefix for Elekta or CTF''')
    parser.add_argument('-mains_freq',
                      help='''Mains frequency for the electrical power noize.
                      Defaults to 60Hz''', default=60, type=int)
    args = parser.parse_args()
    
    filename = args.filename

    if args.output_base:
        calc_ica(filename, outbasename=args.output_base, 
                 mains_freq=args.mains_freq)
    else:
        calc_ica(filename, mains_freq=args.mains_freq)

