#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:29:36 2024

@author: jstout
"""
import os, os.path as op
from enigmeg.process_meg import process
import mne
from enigmeg.process_meg import write_aparc_sub
import pytest
import pygit2
from numpy import allclose
import numpy as np


enigma_test_dir = op.join(os.environ['ENIGMA_TEST_DIR'], 'enigma_test_data')
#repo = pygit2.Repository(enigma_test_dir) 
#TODO - force repo back to original hash to offset any processing changes
#repo.checkout(refname=......)

# =============================================================================
# CAMCAN
# =============================================================================
os.environ['n_jobs']='6'
elekta1_kwargs = {'subject':'CC110101',
              'bids_root':op.join(enigma_test_dir, 'CAMCAN'), 
              'run':'01',
              'session':'01',
              'mains':50,
              'rest_tagname':'rest',
              'emptyroom_tagname':'emptyroom',
              }

ctf1_kwargs=   {'subject':'A2021',
              'bids_root':op.join(enigma_test_dir, 'MOUS'), 
              'run': None,
              'session': None,
              'mains':50,
              'rest_tagname':'rest',
              'emptyroom_tagname': None,
              } 

fourD1_kwargs=   {'subject':'100307',
              'bids_root':op.join(enigma_test_dir, 'HCP'), 
              'run': '01',
              'session': '1',
              'mains':60,
              'rest_tagname':'rest',
              'emptyroom_tagname': 'empty',
              } 

@pytest.mark.parametrize("kwargs", [elekta1_kwargs, ctf1_kwargs, fourD1_kwargs])
def test_vendor_proc(kwargs):
    proc=process(**kwargs)
    proc.load_data()
    #Crop data for testing
    proc.raw_rest.crop(0, 180)
    proc.raw_eroom.crop(0, 180)
    
    proc.vendor_prep(megin_ignore=proc._megin_ignore)
    proc.do_ica()
    proc.do_classify_ica()
    proc.do_preproc()
    proc.do_clean_ica()
    proc.do_proc_epochs()
    proc.proc_mri(t1_override=proc._t1_override)
    proc.do_beamformer()
    proc.do_make_aparc_sub()
    proc.do_label_psds()
    proc.do_spectral_parameterization()
    proc.do_mri_segstats()
    proc.cleanup()

#%%

    
# def test_elekta1_outputs():
#     #Setup
#     bids_root = elekta1_kwargs['bids_root']
#     deriv_root = op.join(bids_root, 'derivatives')
#     enigma_root = op.join(deriv_root, 'ENIGMA_MEG')
#     out_meg_root = op.join(enigma_root, 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
#     gt_meg_root = op.join(enigma_test_dir, 'all_derivatives', 'CAMCAN', 'ENIGMA_MEG', 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
#     target = f'sub-{elekta1_kwargs["subject"]}_ses-01_meg_run-01_headpos.npy'
#     headpos = op.join(out_meg_root, target)
#     headpos_gt = op.join(gt_meg_root, target)
#     assert np_compare(headpos, headpos_gt, tol=.01)    #< Fix    

    
         