#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:30:42 2020

@author: stoutjd
"""


import os
import glob
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.datasets import sample
from mne.simulation.raw import simulate_raw
from mne.simulation.source import SourceSimulator

from neurodsp import sim
from neurodsp.sim import sim_combined
from neurodsp.spectral import compute_spectrum

def return_pseudo_neuro_sig(peakFreq=None, peakBw=None, burstProp=0.3,
                            sigDuration=None, sfreq=None, ooofExp=-2,
                            sigAmplitudeNam=None):
    '''Fix - scale to a specified amount of neural power'''
    components = {'sim_bursty_oscillation' : {'freq' : peakFreq,
                                              'enter_burst' : burstProp},
                  'sim_powerlaw' : {'exponent' : ooofExp, 'f_range' : (1, None)}}
    signal = sim_combined(sigDuration, sfreq, components)
    return signal * sigAmplitudeNam * 1e-9


def generate_subjects_psuedomeg(subjid=None, 
                                subjects_dir=None, 
                                raw_fname=None,
                                trans_fname=None, 
                                bem_fname=None,
                                src_fname=None,
                                sfreq=400,
                                duration=10, 
                                input_dir=None, 
                                get_hcp=None):
    if subjects_dir==None:
        try:
            subjects_dir=os.environ['SUBJECTS_DIR']
        except:
            print('SUBJECTS_DIR not defined in os.environ or commandline')
            raise(ValueError)

    src_fname = glob.glob(op.join(input_dir, subjid, '*-src.fif'))[0]
    src = mne.read_source_spaces(src_fname)
    bem_fname = glob.glob(op.join(input_dir, subjid, '*-sol.fif'))[0]
    bem = mne.read_bem_solution(bem_fname)
    trans = mne.read_trans(trans_fname)

    
    if get_hcp !=None:
        raw, eroom=read_hcp_input()
        info = raw.info
    elif raw_fname[-3:]=='.ds':
        info = mne.io.read_raw_ctf(raw_fname, clean_names=True).info
    elif raw_fname[-4:]=='.fif':
        info = mne.io.read_raw_fif(raw_fname).info
    info.update(sfreq=sfreq, bads=[])
    
    fwd = mne.make_forward_solution(info=info, trans=trans,src=src, 
                                    bem=bem_fname, meg=True)
    
    labels=mne.read_labels_from_annot(subjid, 
                                      subjects_dir=subjects_dir)

    source_simulator = SourceSimulator(src, tstep=1/sfreq, duration=duration)
    
    #Generate uniform distribution of alpha peaks between 8 and 12
    alpha_rand = np.random.uniform(8, 12, size=len(labels))
    
    dframe=pd.DataFrame()
    
    for idx, label in enumerate(labels):
        np.random.seed(idx)
        sig=return_pseudo_neuro_sig(peakFreq=alpha_rand[idx], peakBw=2, burstProp=0.3,
                                sigDuration=duration, sfreq=sfreq, ooofExp=-2,
                                sigAmplitudeNam=1)
        np.save(label.name+'_sig.npy', sig)
        source_simulator.add_data(label, sig, [[0, 0, 1]])
        
        dframe.loc[idx, 'label']=label.name
        dframe.loc[idx, 'seed']=idx
        dframe.loc[idx, 'alpha_val'] = alpha_rand[idx]
    
    dframe.to_csv('./simulation_values.csv', index=False)

    #FIX  - For CTF files, the simulation does not apply to ref data
    if raw_fname[-3:]=='.ds':
        ch_names = [i for i in info.ch_names if len(i)==5]
        info.pick_channels(ch_names)

    raw = simulate_raw(info, source_simulator, forward=fwd)
    raw.save('{}_NeuroDSP_sim_meg.fif'.format(subjid))
    
# def test_hcp_sim():
#     #os.chdir('/home/stoutjd/src/enigma/tmp')
#     #data_path = os.path.join(os.path.realpath(__file__), '../../test_data')
#     subjid = 'AYCYELJY_fs'
#     enigma_outputs = os.path.join(data_path, 'enigma_outputs')
#     subjects_dir= os.path.join(data_path, 'SUBJECTS_DIR') #, subjid) 
#     raw_fname = os.path.join(data_path, 'HCP', 'hcp_rest_example.fif')
#     trans_fname=os.path.join(data_path, 'CTF', 'ctf-trans.fif') 
#     bem_fname=os.path.join(enigma_outputs, subjid, 'bem_sol-sol.fif')
#     src_fname=os.path.join(enigma_outputs, subjid, 'source_space-src.fif')
#     sfreq=100
#     duration=10 
#     generate_subjects_psuedomeg(subjid=subjid, 
#                                 subjects_dir=subjects_dir, 
#                                 raw_fname=raw_fname,
#                                 trans_fname=trans_fname, 
#                                 bem_fname=bem_fname,
#                                 src_fname=src_fname,
#                                 sfreq=100,
#                                 duration=10, 
#                                 input_dir=enigma_outputs, 
#                                 get_hcp=None)
    
    
# def read_hcp_input():
#     '''Returns test data for hcp
#     raw, errom =  return_hcp_test_files()'''
#     hcp_fname = '/home/stoutjd/src/enigma/test_data/HCP/hcp_rest_example.fif'
#     eroom_fname = '/home/stoutjd/src/enigma/test_data/HCP/hcp_eroom_example.fif'
#     raw = mne.io.read_raw_fif(hcp_fname)
#     eroom = mne.io.read_raw_fif(eroom_fname)
#     return raw, eroom



# def generate_sine_signal(subjid=None, 
#                                 subjects_dir=None, 
#                                 raw_fname=None,
#                                 trans_fname=None, 
#                                 bem_fname=None,
#                                 src_fname=None,
#                                 sfreq=400,
#                                 duration=10, 
#                                 input_dir=None, 
#                                 get_hcp=None):
#     if subjects_dir==None:
#         try:
#             subjects_dir=os.environ['SUBJECTS_DIR']
#         except:
#             print('SUBJECTS_DIR not defined in os.environ or commandline')
#             raise(ValueError)

#     src_fname = glob.glob(op.join(input_dir, subjid, '*-src.fif'))[0]
#     src = mne.read_source_spaces(src_fname)
#     bem_fname = glob.glob(op.join(input_dir, subjid, '*-sol.fif'))[0]
#     bem = mne.read_bem_solution(bem_fname)
#     trans = mne.read_trans(trans_fname)

    
#     if get_hcp !=None:
#         raw, eroom=read_hcp_input()
#         info = raw.info
#     elif raw_fname[-3:]=='.ds':
#         info = mne.io.read_raw_ctf(raw_fname, clean_names=True).info
#     elif raw_fname[-4:]=='.fif':
#         info = mne.io.read_raw_fif(raw_fname).info
#     info.update(sfreq=sfreq, bads=[])
    
#     fwd = mne.make_forward_solution(info=info, trans=trans,src=src, 
#                                     bem=bem_fname, meg=True)
    
#     labels=mne.read_labels_from_annot(subjid, 
#                                       subjects_dir=subjects_dir)

#     source_simulator = SourceSimulator(src, tstep=1/sfreq, duration=duration)
    
#     for idx, label in enumerate(labels):
#     np.random.seed(idx)
#     sig=dat_fun(raw.times)
    
#     np.save(label.name+'_sig.npy', sig)
#     source_simulator.add_data(label, sig, [[0, 0, 1]])
#     return source_simulator
   


    
    
    
def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz"""
    """From https://mne.tools/stable/auto_examples/simulation/plot_simulate_raw_data.html#sphx-glr-auto-examples-simulation-plot-simulate-raw-data-py"""
    # n_samp = len(times)
    data = 25e-9 * np.sin(2. * np.pi * 10. * times)
    return data

# def test_generate_sine_signal():
#     from enigmeg.test_data.get_test_data import datasets
#     test_dat = datasets().elekta
    
#     sig = generate_sine_signal(subjid=test_dat['subject'], 
#                         subjects_dir=test_dat['SUBJECTS_DIR'], 
#                         raw_fname=test_dat['meg_rest'],
#                         trans_fname=test_dat['trans'], 
#                         bem_fname=test_dat['bem'],
#                         src_fname=test_dat['src'],
#                         sfreq=400,
#                         duration=10, 
#                         input_dir=test_dat['enigma_outputs'], 
#                         get_hcp=None)
    
#     #FIX  - For CTF files, the simulation does not apply to ref data
#     if raw_fname[-3:]=='.ds':
#         ch_names = [i for i in info.ch_names if len(i)==5]
#         info.pick_channels(ch_names)

#     raw = simulate_raw(info, source_simulator, forward=fwd)
#     raw.save('{}_NeuroDSP_sim_meg.fif'.format(subjid))
    

    

def compare_simulation_signals(sim_dir=None):
    '''Evaluate the ground truth simulated data to the recovered signal'''
    dframe = pd.read_csv(op.join(sim_dir, 'simulation_values.csv'))
    
    for idx,row in dframe.iterrows():
        # print(row)
        print(row['label'])
        current = np.load(op.join(sim_dir, row['label']+'_sig.npy'))
        recovered = None
        
        print(len(current))
        
                          
    
    labels=mne.read_labels_from_annot(subjid, 
                                  subjects_dir=subjects_dir)
    

    
    
if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjid', help='Subject ID in Freesurfer')
    parser.add_argument('-subjects_dir', help='')
    parser.add_argument('-megfile', help='meg file to pull the info')
    parser.add_argument('-derivatives_dir', help='location of src, bem, trans')
    parser.add_argument('-duration', help='Duration of file in seconds', type=float)
    parser.add_argument('-sfreq', help='Sampling frequency', type=float)
    parser.add_argument('-transfile', help='MNE transformation file')
    
    args=parser.parse_args()
    
    generate_subjects_psuedomeg(subjid=args.subjid,
                                subjects_dir=args.subjects_dir,
                                raw_fname=args.megfile,
                                sfreq=args.sfreq,
                                duration=args.duration, 
                                trans_fname=args.transfile,
                                input_dir=args.derivatives_dir)
    



        
