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



enigma_test_dir = op.join(os.environ['ENIGMA_TEST_DIR'], 'enigma_test_data')
repo = pygit2.Repository(enigma_test_dir) 
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

def test_elekta1_proc(kwargs):
    proc=process(**elekta1_kwargs)
    proc.load_data()
    # assert proc.
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

# =============================================================================
# MOUS - CTF    
# =============================================================================
ctf1_kwargs=   {'subject':'A2021',
              'bids_root':op.join(enigma_test_dir, 'MOUS'), 
              'run': None,
              'session': None,
              'mains':50,
              'rest_tagname':'rest',
              'emptyroom_tagname': None,
              }  
def test_ctf1_proc(): 
    proc=process(**ctf1_kwargs)
    proc.load_data()
    # assert proc.
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





# =============================================================================
# HCP - 4D
# =============================================================================
hcp1_kwargs=   {'subject':'100307',
              'bids_root':op.join(enigma_test_dir, 'HCP'), 
              'run': '01',
              'session': '1',
              'mains':60,
              'rest_tagname':'rest',
              'emptyroom_tagname': 'empty',
              } 

def test_hcp1_proc(): 
    proc=process(**hcp1_kwargs)
    proc.load_data()
    # assert proc.
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
          

