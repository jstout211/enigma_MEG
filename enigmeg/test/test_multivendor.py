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

def get_fname(topdir, tag=None):
    tmp = glob.glob(op.join(topdir, '**',f'*{tag}.fif'), recursive=True)
    if len(tmp) > 1:
        raise ValueError(f'More than one {tag} files')
    return tmp[0]













# @pytest.mark.parametrize("kwargs", [elekta1_kwargs, ctf1_kwargs, fourD1_kwargs])        
# def test_preproc(kwargs):
#     sub-CC110101_ses-01_run-01_proc-mcorr_meg.fif
#     sub-CC110101_ses-01_task-emptyroom_epo.fif
#     sub-CC110101_ses-01_task-emptyroom_proc-filt_meg.fif
#     sub-CC110101_ses-01_task-rest_run-01_ica/
#     sub-CC110101_ses-01_task-rest_run-01_epo.fif
#     sub-CC110101_ses-01_task-rest_run-01_proc-filt_meg.fif
# sub-CC110101_ses-01_meg_run-01_headpos.npy
    
# @pytest.mark.parametrize("kwargs", [elekta1_kwargs, ctf1_kwargs, fourD1_kwargs])        
# def test_fooof_outputs(kwargs):
#     sub-CC110101_ses-01_meg_band_task-rest_run-01_rel_power.csv
#     sub-CC110101_ses-01_meg_label_task-rest_run-01_spectra.csv    

# @pytest.mark.parametrize("kwargs", [elekta1_kwargs, ctf1_kwargs, fourD1_kwargs])        
# def test_source_loc_outputs(kwargs):
#     #Transform
#     trans_fname = 
#     gt_trans_fname = 
#     trans = 
#     gt_trans = 
#     sub-CC110101_ses-01_task-rest_run-01_trans.fif
    
#     #Covariance
#     restcov_fname = 
#     gt_restcov_fname = 
#     restcov = 
#     gt_restcov = 
    
#     #Beamformer Filters
#     filters = 
#     gt_filters = 
#     sub-CC110101_ses-01_run-01_lcmv.h5
    
#     sub-CC110101_ses-01_task-emptyroom_cov.fif
#     sub-CC110101_ses-01_task-rest_run-01_cov.fif
    

    


@pytest.mark.parametrize("kwargs", [elekta1_kwargs, ctf1_kwargs, fourD1_kwargs])    
def test_anat_outputs(kwargs):
    bids_root = kwargs['bids_root']
    deriv_root = op.join(bids_root, 'derivatives')
    enigma_root = op.join(deriv_root, 'ENIGMA_MEG')
    repo_name = op.basename(bids_root)
    gt_meg_root = op.join(enigma_test_dir, 'all_derivatives', f'{repo_name}_crop', 'ENIGMA_MEG')

    # Source space test
    # Left Hemi
    src_fname = get_fname(enigma_root, tag='src')
    gt_src_fname = get_fname(gt_meg_root, tag='src')
    src = mne.read_source_spaces(src_fname)[0]
    gt_src = mne.read_source_spaces(gt_src_fname)[0]
    assert allclose(src['rr'], gt_src['rr'])
    assert allclose(src['nn'], gt_src['nn'])
    assert allclose(src['vertno'], gt_src['vertno'])
    # Right Hemi
    src = mne.read_source_spaces(src_fname)[1]
    gt_src = mne.read_source_spaces(gt_src_fname)[1]
    assert allclose(src['rr'], gt_src['rr'])
    assert allclose(src['nn'], gt_src['nn'])
    assert allclose(src['vertno'], gt_src['vertno'])
    
    # BEM test
    bem_fname = get_fname(enigma_root, tag='bem')
    gt_bem_fname =  get_fname(gt_meg_root, tag='bem')
    bem = mne.bem.read_bem_solution(bem_fname)
    gt_bem = mne.bem.read_bem_solution(gt_bem_fname)
    assert type(bem) == type(gt_bem)
    assert allclose(bem['solution'], gt_bem['solution'])
    
    # Forward Model 
    # fwd_fname = 
    # gt_fwd_fname = 
    # fwd = 
    # gt_fwd = 
    # sub-CC110101_ses-01_task-rest_run-01_fwd.fif
    
    # Parcellation   FIX - should test this
    # parc_fname = op.join(deriv_root, 'freesurfer', 'subjects',f'sub-{kwargs["subject"]}', 'label','lh.aparc_sub.annot')
    # gt_parc_fname = op.join(deriv_root, 'freesurfer', 'subjects',f'sub-{kwargs["subject"]}', 'label','lh.aparc_sub.annot')
    # parc = 
    # gt_parc = 

@pytest.mark.parametrize("kwargs", [elekta1_kwargs, ctf1_kwargs, fourD1_kwargs])    
def test_logfile(kwargs):
    #Setup
    bids_root = kwargs['bids_root']
    deriv_root = op.join(bids_root, 'derivatives')
    enigma_root = op.join(deriv_root, 'ENIGMA_MEG')
    repo_name = op.basename(bids_root)
    out_meg_root = op.join(enigma_root, 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
    gt_meg_root = op.join(enigma_test_dir, 'all_derivatives', f'{repo_name}_crop', 'ENIGMA_MEG', 'sub-'+elekta1_kwargs['subject'], 'ses-01','meg')
    
    #Logfile testings
    subject, run, session, rest_tagname = kwargs['subject'], kwargs['run'], kwargs['session'], kwargs['rest_tagname']
    
    logfile_fname = op.join(enigma_root, 'logs',f'{subject}_ses-{session}_task-{rest_tagname}_run-{run}_log.txt')
    gt_logfile_fname = op.join(gt_meg_root, 'logs', f'{subject}_ses-{session}_task-{rest_tagname}_run-{run}_log.txt')
    
    with open(logfile_fname) as f:
        logfile = f.readlines()
        logfile = [i.split('INFO')[-1] for i in logfile]
    with open(gt_logfile_fname) as f:
        gt_logfile = f.readlines()
        gt_logfile = [i.split('INFO')[-1] for i in gt_logfile]

    for i,j in zip(logfile, gt_logfile):
        assert i==j


    
    
    
         