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
import pandas as pd
# import pygit2
from numpy import allclose
import numpy as np
import datalad as dl
import glob

enigma_test_dir = op.join(os.environ['ENIGMA_TEST_DIR'], 'enigma_test_data')

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

kit1_kwargs = {'subject' : '0001',
               'bids_root': op.join(enigma_test_dir, 'YOKOGOWA'), 
               'run': '1',
               'session': '1',
               'mains':50,
               'rest_tagname':'eyesclosed',
               'emptyroom_tagname': None
               }

kwarg_list = [elekta1_kwargs, ctf1_kwargs, kit1_kwargs]
all_vendors = pytest.mark.parametrize("kwargs", kwarg_list,  ids=['MEGIN', 'CTF', 'KIT'])

@all_vendors
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
    

#%% Helper functions
def get_fname(topdir, tag=None):
    tmp = glob.glob(op.join(topdir, '**',f'*{tag}.fif'), recursive=True)
    if len(tmp) > 1:
        raise ValueError(f'More than one {tag} files')
    return tmp[0]

def get_dirs(kwargs):
    bids_root = kwargs['bids_root']
    repo_name = op.basename(bids_root)
    deriv_root = op.join(bids_root, 'derivatives')
    enigma_root = op.join(deriv_root, 'ENIGMA_MEG')
    gt_enigma_root = op.join(enigma_test_dir, 'all_derivatives', f'{repo_name}_crop', 'ENIGMA_MEG')
    return bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name

#%% Preprocessing

@pytest.mark.parametrize("kwargs", [elekta1_kwargs], ids=['MEGIN'])
def test_mcorr_outputs(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)
    
    #Test
    target = f'sub-{kwargs["subject"]}_ses-01_meg_run-01_headpos.npy'
    headpos_fname = glob.glob(op.join(enigma_root, '**', target), recursive=True)[0]
    gt_headpos_fname = glob.glob(op.join(gt_enigma_root, '**', target), recursive=True)[0]
    headpos = np.load(headpos_fname)
    headpos_gt = np.load(gt_headpos_fname)
    assert allclose(headpos, headpos_gt, atol=1e-3)


#%% Anatomical
@all_vendors
def test_src(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)

    # Source space test
    # Left Hemi
    src_fname = get_fname(enigma_root, tag='src')
    gt_src_fname = get_fname(gt_enigma_root, tag='src')
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


@all_vendors
def test_bem(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)
    
    bem_fname = get_fname(enigma_root, tag='bem')
    gt_bem_fname =  get_fname(gt_enigma_root, tag='bem')
    bem = mne.bem.read_bem_solution(bem_fname)
    gt_bem = mne.bem.read_bem_solution(gt_bem_fname)
    assert type(bem) == type(gt_bem)
    assert allclose(bem['solution'], gt_bem['solution'])

@all_vendors
def test_fwd(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)
    
    fwd_fname = get_fname(enigma_root, tag='fwd')
    gt_fwd_fname = get_fname(gt_enigma_root, tag='fwd')
    fwd = mne.read_forward_solution(fwd_fname)
    gt_fwd = mne.read_forward_solution(gt_fwd_fname)
    assert allclose(fwd['sol']['data'], gt_fwd['sol']['data'])

@all_vendors
def test_aparcsub(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)
    
    for HEMI in {'lh','rh'}:
        fs_dir = op.join(deriv_root, 'freesurfer', 'subjects')
        gt_fs_dir = op.join(deriv_root, 'freesurfer', 'subjects')
        HEMI='lh'
        parc = mne.read_labels_from_annot(f'sub-{kwargs["subject"]}',
                                          parc='aparc_sub', 
                                          subjects_dir=fs_dir, 
                                          hemi=HEMI)
        gt_parc = mne.read_labels_from_annot(f'sub-{kwargs["subject"]}',
                                          parc='aparc_sub', 
                                          subjects_dir=gt_fs_dir, 
                                          hemi=HEMI)
        for i,j in zip(parc, gt_parc):
            assert i.name == j.name
            assert np.alltrue(i.vertices == j.vertices)
            

#%% Source localization
@all_vendors
def test_transform(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)

    trans_fname = get_fname(enigma_root, tag='trans')
    gt_trans_fname = get_fname(gt_enigma_root, tag='trans')
    trans = mne.read_trans(trans_fname)
    gt_trans = mne.read_trans(gt_trans_fname)
    assert allclose(trans['trans'], gt_trans['trans'])
    assert trans['to']==gt_trans['to']
    assert trans['from']==gt_trans['from'] 

@all_vendors
def test_covariance(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)

    restcov_fname =  glob.glob(op.join(enigma_root, '**','*rest*cov*.fif'), recursive=True)[0]
    gt_restcov_fname = glob.glob(op.join(gt_enigma_root, '**','*rest*cov*.fif'), recursive=True)[0]
    restcov = mne.read_cov(restcov_fname)
    gt_restcov = mne.read_cov(gt_restcov_fname)
    assert np.allclose(restcov['data'], gt_restcov['data'])    

@all_vendors
def test_beamformer(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)

    beam_fname = glob.glob(op.join(enigma_root, '**','*lcmv.h5'), recursive=True)[0]
    gt_beam_fname =  glob.glob(op.join(gt_enigma_root, '**','*lcmv.h5'), recursive=True)[0]
    beam = mne.beamformer.read_beamformer(beam_fname)
    gt_beam = mne.beamformer.read_beamformer(gt_beam_fname)
    assert np.allclose(beam['weights'], gt_beam['weights'], rtol=0.01)

@all_vendors
def test_logfile(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)

    #Logfile testings
    subject, run, session, rest_tagname = kwargs['subject'], kwargs['run'], kwargs['session'], kwargs['rest_tagname']
    
    logfile_fname = glob.glob(op.join(enigma_root, 'logs',f'{subject}_*_log.txt'))[0]
    gt_logfile_fname = glob.glob(op.join(gt_enigma_root, 'logs', f'{subject}_*_log.txt'))[0]
    
    with open(logfile_fname) as f:
        logfile = f.readlines()
        logfile = [i.split('INFO')[-1] for i in logfile]
    with open(gt_logfile_fname) as f:
        gt_logfile = f.readlines()
        gt_logfile = [i.split('INFO')[-1] for i in gt_logfile]

    for i,j in zip(logfile, gt_logfile):
        assert i==j



#%% Test final outputs
@all_vendors
def test_spectra_outputs(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)
    
    #Test
    spectra_fname = glob.glob(op.join(enigma_root, f'sub-{kwargs["subject"]}', '**', '*spectra.csv'), recursive=True)[0]
    gt_spectra_fname = glob.glob(op.join(gt_enigma_root, f'sub-{kwargs["subject"]}', '**', '*spectra.csv'), recursive=True)[0]
    spectra = pd.read_csv(spectra_fname)    
    gt_spectra = pd.read_csv(gt_spectra_fname)
    assert np.allclose(spectra.values, gt_spectra.values, atol=0.0001) 

@all_vendors
def test_fooof_outputs(kwargs):
    bids_root, deriv_root, enigma_root, gt_enigma_root, repo_name = get_dirs(kwargs)
    
    #Test
    relpow_fname = glob.glob(op.join(enigma_root, f'sub-{kwargs["subject"]}', '**', '*rel_power.csv'), recursive=True)[0]
    gt_relpow_fname = glob.glob(op.join(gt_enigma_root, f'sub-{kwargs["subject"]}', '**', '*rel_power.csv'), recursive=True)[0]
    relpow = pd.read_csv(relpow_fname, sep='\t', index_col=0)
    gt_relpow = pd.read_csv(gt_relpow_fname, sep='\t', index_col=0)
    assert np.allclose(relpow.values, gt_relpow.values, atol=0.01)

    
    
    
         
