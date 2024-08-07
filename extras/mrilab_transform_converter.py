#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:08:49 2024

@author: jstout
"""
import mne
import os
import numpy as np


def convert_mrilab_trans(fname=None, subject=None):
    showfiff_result_fname =  fname
    with open(showfiff_result_fname) as f:
        showfiff_results = f.readlines()
        
    assert showfiff_results[0] == '222 = transform      \thead -> MRI\n'
    showfiff_results = [i.strip('\t').strip('\n') for i in showfiff_results]
    rot = np.array([i.split() for i in showfiff_results[1:4]] , dtype=float)
    trans_mat = np.zeros([4,4])
    trans_mat[:3, :3] = rot
    
    translation = np.array(showfiff_results[4].split()[0:3], dtype=float)
    
    trans_mat[:3, 3] = translation.T / 1000
    trans_mat[3,3]=1
    
    trans = mne.Transform('head', 'mri', trans=trans_mat)
    trans.save(f'{subject}_mne_trans.fif')
    

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert ')
    parser.add_argument('-subject', 
                        help='Subject ID to save the resulting trans.fif file'
                        )
    parser.add_argument('-fname', 
                        help = '''Enter the filename of the output text file 
                        produced by show_fif --tag 222 --verbose --in {MRILAB_Coreg.fif} > {ShowFifOutput}.txt'''
                        )
    args = parser.parse_args()
    convert_mrilab_trans(fname = args.fname, subject=args.subject)
    