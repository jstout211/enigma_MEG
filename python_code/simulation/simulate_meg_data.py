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
                                input_dir=None):
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

    info = mne.io.read_raw_ctf(raw_fname, clean_names=True).info
    info.update(sfreq=sfreq, bads=[])
    
    fwd = mne.make_forward_solution(info=info, trans=trans,src=src, 
                                    bem=bem_fname, meg=True)
    
    labels=mne.read_labels_from_annot(subjid, 
                                      subjects_dir=subjects_dir)

    source_simulator = SourceSimulator(src, tstep=1/sfreq, duration=duration)
    
    for rseed, label in enumerate(labels[0:10]):
        np.random.seed(rseed)
        sig=return_pseudo_neuro_sig(peakFreq=10, peakBw=2, burstProp=0.3,
                                sigDuration=duration, sfreq=sfreq, ooofExp=-2,
                                sigAmplitudeNam=1)
        np.save(label.name+'_sig.npy', sig)
        source_simulator.add_data(label, sig, [[0, 0, 1]])




    #FIX  - ONly works for CTF currently #####################################
    #################
    ch_names = [i for i in info.ch_names if len(i)==5]
    info.pick_channels(ch_names)

    raw = simulate_raw(info, source_simulator, forward=fwd)
    raw.save('{}_NeuroDSP_sim_meg.fif'.format(subjid))
    
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
    



        
