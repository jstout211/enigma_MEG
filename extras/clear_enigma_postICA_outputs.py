#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:33:48 2026

@author: jstout
"""

import os, os.path as op
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-bids_root')
args = parser.parse_args()



#% Setup
bids_root = args.bids_root
enigma_deriv = op.join(bids_root, 'derivatives', 'ENIGMA_MEG')
outdir = op.join(bids_root, 'derivatives', 'ENIGMA_MEG_OLD')

assert op.exists(enigma_deriv)
if not op.exists(outdir): os.mkdir(outdir)

#%
rm_list = ['*_rel_power.csv', 
           '*_spectra.csv',
           '*_lcmv.h5',
           '*_cov.fif', 
           '*_epo.fif',
           ]

subj_list = glob.glob('sub-*', root_dir=enigma_deriv) #bids_root)

for subject in subj_list:
    for item in rm_list:
        found_item = glob.glob(f'{enigma_deriv}/{subject}/**/meg/{item}', recursive=True)
        for _i  in found_item:
            outfile = _i.replace(enigma_deriv, outdir)
            _new_dir = op.dirname(outfile)
            if not op.exists(_new_dir): os.makedirs(_new_dir)
            print(f'Moving {_i} to {outfile}')
            shutil.move(_i, outfile)
    
           
           