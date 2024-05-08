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
# import pygit2
from numpy import allclose
import numpy as np
import datalad as dl

enigma_test_dir = op.join(os.environ['ENIGMA_TEST_DIR'], 'enigma_test_data')
#repo = pygit2.Repository(enigma_test_dir) 
#TODO - force repo back to original hash to offset any processing changes
#repo.checkout(refname=......)

# =============================================================================
# Data Prep
# =============================================================================
# Remove label files and unlock label folder
# rm op.join(subjects_dir, 'morph_maps')
# Remove 



# =============================================================================
# Test Setups
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
    if proc.raw_eroom != None:
        if proc.raw_eroom.times[-1] > 180:
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

#%% Test results
# def anat_list(prefix=None, suffix=None):
#     outlist = {}
#     outlist{'bem'}=f'{prefix}_bem.fif'
#     outlist{'src'}=f'{prefix}_src.fif'
#     outlist{
#     bem_fname=

@pytest.mark.parametrize("kwargs", [elekta1_kwargs]) #, ctf1_kwargs, fourD1_kwargs])
def test_elekta1_outputs(kwargs):
    #Setup
    bids_root = kwargs['bids_root']
    deriv_root = op.join(bids_root, 'derivatives')
    enigma_root = op.join(deriv_root, 'ENIGMA_MEG')
    out_meg_root = op.join(enigma_root, 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
    gt_meg_root = op.join(enigma_test_dir, 'all_derivatives', 'CAMCAN_crop', 'ENIGMA_MEG', 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
    target = f'sub-{elekta1_kwargs["subject"]}_ses-01_meg_run-01_headpos.npy'
    #Movement corr
    headpos = np.load(op.join(out_meg_root, target))
    headpos_gt = np.load(op.join(gt_meg_root, target))
    assert allclose(headpos, headpos_gt, atol=1e-3)
    #Finish HERE ---- 
    
def test_logfile(elekta_kwargs):
    kwargs = elekta1_kwargs  #!!!!!!!!!!! HACK - hard coded FIX
    #Setup
    bids_root = kwargs['bids_root']
    deriv_root = op.join(bids_root, 'derivatives')
    enigma_root = op.join(deriv_root, 'ENIGMA_MEG')
    out_meg_root = op.join(enigma_root, 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
    gt_meg_root = op.join(enigma_test_dir, 'all_derivatives', 'CAMCAN_crop', 'ENIGMA_MEG', 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
    
    #Logfile testings
    subject, run, session, rest_tagname = kwargs['subject'], kwargs['run'], kwargs['session'], kwargs['rest_tagname']
    
    logfile_fname = op.join(enigma_root, 'logs',f'{subject}_ses-{session}_task-{rest_tagname}_run-{run}_log.txt')
    gt_logfile_fname = op.join(op.dirname(elekta1_kwargs['bids_root']), 'all_derivatives', 'CC110101_ses-01_task-rest_run-01_log.txt')
    
    with open(logfile_fname) as f:
        logfile = f.readlines()
        logfile = [i.split('INFO')[-1] for i in logfile]
    with open(gt_logfile_fname) as f:
        gt_logfile = f.readlines()
        gt_logfile = [i.split('INFO')[-1] for i in gt_logfile]
    
    for i,j in zip(logfile, gt_logfile):
        assert i==j
        
        
    
    
    
    
         