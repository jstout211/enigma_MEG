#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:39:25 2023

@author: Allison Nugent and Jeff Stout
"""

import os, os.path as op
import argparse
from enigmeg.process_meg import process
from enigmeg.QA.enigma_QA_functions import gen_coreg_pngs, gen_bem_pngs, gen_src_pngs, gen_surf_pngs
from enigmeg.QA.enigma_QA_functions import gen_epo_pngs, gen_fooof_pngs
import sys

def _prepare_QA(subjstruct):
    
    gen_coreg_pngs(subjstruct)

    gen_bem_pngs(subjstruct)

    gen_src_pngs(subjstruct)

    gen_surf_pngs(subjstruct)
    
    gen_epo_pngs(subjstruct)
    
    gen_fooof_pngs(subjstruct)

if __name__=='__main__':
    
    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-topdir', help='''top dir containing bids root directory''')
    parser.add_argument('-bids_root', help='''BIDS root directory''')
    parser.add_argument('-subjects_dir', help='''Freesurfer subjects_dir can be assigned at the commandline if not already exported''')
    parser.add_argument('-subjid', help='''Define the subject id to process''')
    parser.add_argument('-run', help='''Run number, note that 01 is different from 1''', default='1')
    parser.add_argument('-session', help='''Session number''', default=None)
    parser.description='''This python script will compile a series of QA images for assessment of the enigma_MEG pipeline'''
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(1)   
    
    if args.topdir:
        topdir=args.topdir
    else:
        topdir=os.getcwd()
    
    if not args.bids_root:
        bids_root = op.join(topdir, 'bids_out')
    else:
        bids_root=args.bids_root
        
    if not op.exists(bids_root):
        raise ValueError('Please specify a correct -bids_root')
    
    if not args.subjid:
        raise ValueError('Must give a subject id')
    else:
        subjid=args.subjid
        
    derivatives_dir = op.join(bids_root, 'derivatives')
    enigma_root = op.join(derivatives_dir, 'ENIGMA_MEG')
    
    if not op.exists(enigma_root):
        raise ValueError('No ENIGMA_MEG directory - did you run process_meg.py?')
                
    if args.subjects_dir == None:
        subjects_dir = op.join(derivatives_dir,'freesurfer/subjects')
    else:
        subjects_dir = args.subjects_dir    
    
    QA_dir = op.join(derivatives_dir,'ENIGMA_MEG_QA/')
    if not os.path.exists(QA_dir):
        os.mkdir(QA_dir)
        
    png_path=op.join(QA_dir + 'sub-'+subjid)
    if not os.path.exists(png_path):
        os.mkdir(png_path)
    
    subjstruct = process(subject=subjid, 
                        bids_root=bids_root, 
                        deriv_root=derivatives_dir,
                        subjects_dir=subjects_dir,
                        rest_tagname='rest',
                        emptyroom_tagname='emptyroom', 
                        session=args.session, 
                        mains=0,      
                        run=args.run,
                        t1_override=None,
                        fs_ave_fids=False
                        )
    
    _prepare_QA(subjstruct)
    

    
    