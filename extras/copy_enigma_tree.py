#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:39:12 2024

@author: jstout
"""

import shutil
import glob
import os, os.path as op
import argparse



def main(subjct, input_dir, output_dir, copy_fsave):
    #%% REMOTE
    r_bids_root = input_dir
    r_deriv_root = op.join(r_bids_root, 'derivatives')
    r_fs_root = op.join(r_deriv_root, 'freesurfer', 'subjects')
    r_enigma_root = op.join(r_deriv_root, 'ENIGMA_MEG')
    r_enigmaQA_root = op.join(r_deriv_root, 'ENIGMA_MEG_QA')
    
    #%% LOCAL OUTPUTS
    l_bids_root = output_dir
    l_deriv_root = op.join(l_bids_root, 'derivatives')
    l_fs_root = op.join(l_deriv_root, 'freesurfer', 'subjects')
    l_enigma_root = op.join(l_deriv_root, 'ENIGMA_MEG')
    l_enigmaQA_root = op.join(l_deriv_root, 'ENIGMA_MEG_QA')
    
    #%% Build local dirs
    try:
        for dirname in [l_bids_root, l_deriv_root, l_fs_root, l_enigma_root, l_enigmaQA_root]:
            os.makedirs(dirname, exist_ok=True)
    except:
        print(f'Failed to create directory hierarchy, you may not have write permission to {l_bids_root}')
        raise
    
    #%% Copy all subject specific data
    indirs = [r_bids_root, r_fs_root, r_enigma_root, r_enigmaQA_root]
    outdirs = [l_bids_root, l_fs_root, l_enigma_root, l_enigmaQA_root]
     
    for indir,outdir in zip(indirs, outdirs):
        src = op.join(indir, 'sub-'+subject)
        dst = op.join(outdir, 'sub-'+subject)
        print(f'Copying from {src} to {dst}')
        shutil.copytree(src, dst)
    
    for fname in ['participants.tsv','participants.json','README']:
        print(f'Copying {fname}')
        shutil.copy(op.join(r_bids_root, fname), 
                    op.join(l_bids_root, fname)
                    )



if __name__=='__main__':
    parser = argparse.ArgumentParser('Copy a single subjects BIDS data over to new folder')
    parser.add_argument('-subject', default=None,
                        help='Subject ID  (optional).  If left empty - the remote dir will be listed at the prompt')
    parser.add_argument('-rbids', help='Remote Bids directory')
    parser.add_argument('-lbids', help='Local/Destination Bids directory')
    parser.add_argument('-copy_fsave', help='Copy Fsaverage to the freesurfer folder (default=False)', 
                        default=False, action='store_true')
    args = parser.parse_args()
    if args.subject==None:
        subj_list = [op.basename(i) for i in  glob.glob(op.join(args.rbids, 'sub-*'))]
        print(subj_list)
        subject=input('Choose a subject:')
    else:
        subject = args.subject
    if subject[0:4]=='sub-':
        subject=subject[4:]
    input_dir = args.rbids
    output_dir = args.lbids 
    main(subject, input_dir, output_dir, args.copy_fsave)   


