#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: stoutjd
"""

import os, os.path as op
import glob
import mne
from enigmeg import process_meg
import subprocess
import pandas as pd
pd.set_option('display.max_colwidth', 255)
import pytest
import numpy as np

os.environ['n_jobs']='1'



download_path = os.path.expanduser('~')
openneuro_dset='ds004215'
bids_root=op.join(download_path, openneuro_dset)


# =============================================================================
# Build object instance
# =============================================================================
def test_load():
    proc = process_meg.process(subject='ON02747',
                        bids_root=bids_root,
                        session='01',
                        run='01',
                        emptyroom_tagname='noise', 
                        mains=60)
    
    assert proc.check_paths() == None
    proc.load_data()
    assert type(proc.raw_rest) is mne.io.ctf.ctf.RawCTF
    assert type(proc.raw_eroom) is mne.io.ctf.ctf.RawCTF   
    return proc

proc = test_load()

def test_vendor_prep():
    assert proc.vendor_prep() == None
    assert proc.bad_channels != None

def test_preproc():
    assert proc.do_preproc() == None
    
def test_create_epochs():
    assert proc.do_proc_epochs() == None
    assert op.exists(proc.fnames.rest_epo)
    assert op.exists(proc.fnames.rest_cov)
    #assert op.exists(proc.fnames.rest_csd) #add for dics call
    assert op.exists(proc.fnames.eroom_epo)
    assert op.exists(proc.fnames.eroom_cov) 
    #assert op.exists(proc.fnames.eroom_csd) #add for dics call

def test_icaclean():
    assert proc.do_ica() == None
    assert proc.do_classify_ica() == None

def test_mriproc():
    proc.proc_mri() #redo_all=True)
    #assert proc.proc_mri() == None
    assert op.exists(proc.fnames.rest_trans)
    assert op.exists(proc.fnames.bem)
    assert op.exists(proc.fnames.rest_fwd)
    assert op.exists(proc.fnames.src)

#Fails because of SSL error
#def test_aparc():
#    proc.do_make_aparc_sub()
#    assert op.exists(proc.fnames.parc)
                      
def test_beamformer():
    if op.exists(proc.fnames.lcmv):
        os.remove(proc.fnames.lcmv)
    proc.do_beamformer()
    assert op.exists(proc.fnames.lcmv)

#    if op.exists(proc.fnames.dics):  #this will be .lcmv
#        os.remove(proc.fnames.dics)
#    proc.do_beamformer()
#    assert op.exists(proc.fnames.dics)
    
def test_label_psds():
    proc.load_data()
    proc.proc_mri()
    proc.do_beamformer()
    #assert hasattr(proc, 'psds') 
    #assert hasattr(proc, 'freqs')
    #assert op.exists(proc.fnames.dics)
    
    # 'Reduce the epoch count to 3 for compute purposes'
    tmp = []
    for i in 0,1,2:
        tmp.append(next(proc.stcs))
    proc.stcs = tmp
    proc.do_label_psds()
    assert hasattr(proc, 'label_ts')
    
def test_do_spectral_param():
    assert proc.do_spectral_parameterization() == None
    assert op.exists(proc.fnames.spectra_csv) 

#%%  Test CSV input
test_id = 'ON02747'

def test_parse_bids(tmp_path):
    d = tmp_path / "parse_bids"
    d.mkdir()
    out_root = d/'tmp_parse_bids'
    out_root.mkdir()
    os.chdir(out_root)
    cmd_ = f'parse_bids.py -bids_root {bids_root} -rest_tag rest -emptyroom_tag noise'
    subprocess.run(cmd_.split(), check=True)
    csv_file = out_root / 'ParsedBIDS_dataframe.csv'
    assert op.exists(csv_file)
    dframe = pd.read_csv(csv_file, dtype=str)
    idx = dframe[dframe['sub']==test_id].index
    row = dframe.loc[idx]
    assert np.all(row.ses=='01')  
    assert np.all(row.run=='01')
    eroom_gt = op.join(bids_root, 'sub-'+test_id, 'ses-01','meg',f'sub-{test_id}_ses-01_task-noise_run-01_meg.ds')
    assert np.all(row.eroom == eroom_gt) 
    assert np.all(row.type == '.ds')
    meg_gt = op.join(bids_root, 'sub-'+test_id, 'ses-01','meg',f'sub-{test_id}_ses-01_task-rest_run-01_meg.ds')
    assert np.all(row.path == meg_gt)
    #HACK to determine MRI file -- multiple T1w files with same coreg
    mri_gt_dir = op.join(bids_root, 'sub-'+test_id, 'ses-01','anat')
    assert np.all(op.dirname(row.mripath.to_string(index=False)) == mri_gt_dir)
    tmp_ = op.basename(row.mripath.to_string(index=False))
    assert tmp_[0:30]==f'sub-{test_id}_ses-01_acq-MPRAGE_'
    assert tmp_[-10:]=='T1w.nii.gz'


def test_csv_procmeg(tmp_path):
    #Need to make this a session scoped tmp_path - so the setup doesnt need to be redone
    d = tmp_path / "parse_bids"
    d.mkdir()
    out_root = d / 'tmp_parse_bids'
    out_root.mkdir()
    os.chdir(out_root)    
    cmd_ = f'parse_bids.py -bids_root {bids_root} -rest_tag rest -emptyroom_tag noise'
    subprocess.run(cmd_.split(), check=True)
    
    csv_file = out_root / 'ParsedBIDS_dataframe.csv'
    # csv_file = tmp_path / "parse_bids" / "tmp_parse_bids" / "ParsedBIDS_dataframe.csv"
    dframe = pd.read_csv(csv_file, dtype=str)
    idx = dframe[dframe['sub']==test_id].index
    dframe = dframe.loc[idx].reset_index(drop=True)
    dframe.to_csv(csv_file, index=False)
    cmd_ = f'process_meg.py -bids_root {bids_root} -mains 60 -n_jobs 1 -proc_fromcsv {csv_file}'
    tmp_ = subprocess.run(cmd_.split(), check=True)
    

# =============================================================================
# 
# =============================================================================
# import shlex
# import argparse
# class make_args(argparse.Namespace):
#     def __init__(self):
#         self.n_jobs=1
#         self.mains=60.0
#         self.proc_fromcsv=None
    
# args = make_args()
# -bids_root {bids_root} -mains 60 -n_jobs 1 -proc_fromcsv {csv_file}    
    
    

