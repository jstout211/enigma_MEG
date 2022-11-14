#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: stoutjd
"""

import os, os.path as op
import datalad.api as dl
import glob
import mne
from enigmeg import process_meg



download_path = os.path.expanduser('~')
openneuro_dset='ds004215'
downloads=\
    ['sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.json',
      'sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_channels.tsv',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_coordsystem.json',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_meg.ds',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_meg.json',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-noise_run-01_channels.tsv',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-noise_run-01_meg.ds',
      'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-noise_run-01_meg.json']

def get_rest_data(dataset='ds004215',
                  branch='1.0.1',
                  download_location=download_path, 
                  downloads=downloads
                  ):
    '''Retrieve CTF rest data from NIMH HV dataset'''
    dl.install(
        path=op.join(download_path,dataset),
        source=f'https://github.com/OpenNeuroDatasets/{openneuro_dset}.git',
        branch=branch
        )
    curr_dir = os.getcwd()
    os.chdir(op.join(download_path, dataset))    
    dl.get(path=downloads)
    
    os.chdir(curr_dir)    

if not os.path.exists(op.join(download_path,openneuro_dset)):
    get_rest_data(dataset='ds004215',
                      download_location=download_path, 
                      downloads=downloads
                      )

rm_files = ['sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_rec-SCIC_T1w.json',
        'sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_rec-SCIC_T1w.nii.gz']
rm_files = [op.join(os.path.expanduser('~'),openneuro_dset,i) for i in rm_files]
if os.path.exists(rm_files[0]): os.remove(rm_files[0])
if os.path.exists(rm_files[1]): os.remove(rm_files[1])

# =============================================================================
# Build object instance
# =============================================================================
def test_load():
    proc = process_meg.process(subject='ON02747',
                        bids_root=op.join(download_path, openneuro_dset),
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
def test_preproc():
    assert proc.do_preproc() == None
    
def test_create_epochs():
    assert proc.do_proc_epochs() == None
    assert op.exists(proc.fnames.rest_epo)
    assert op.exists(proc.fnames.rest_cov)
    assert op.exists(proc.fnames.eroom_epo)
    assert op.exists(proc.fnames.eroom_cov)

def test_mriproc():
    proc.proc_mri(redo_all=True)
    #assert proc.proc_mri() == None
    assert op.exists(proc.fnames.rest_trans)
    assert op.exists(proc.fnames.bem)
    assert op.exists(proc.fnames.rest_fwd)
    assert op.exists(proc.fnames.src)
    
def test_aparc():
    proc.do_make_aparc_sub()
    # assert op.exists(
                      
def test_beamformer():
    if op.exists(proc.fnames.lcmv):
        os.remove(proc.fnames.lcmv)
    proc.do_beamformer()
    assert op.exists(proc.fnames.lcmv)
    
def test_label_psds():
    proc.load_data()
    proc.proc_mri()
    proc.do_beamformer()
    
    'Reduce the epoch count to 3 for compute purposes'
    tmp = []
    for i in 0,1,2:
        tmp.append(next(proc.stcs))
    proc.stcs = tmp
    proc.do_label_psds()
    assert hasattr(proc, 'label_ts')
    


