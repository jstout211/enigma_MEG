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

os.environ['n_jobs']='1'



download_path = os.path.expanduser('~')
openneuro_dset='ds004215'


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
    if op.exists(proc.fnames.dics):  #this will be .lcmv
        os.remove(proc.fnames.dics)
    proc.do_beamformer()
    assert op.exists(proc.fnames.dics)
    
def test_label_psds():
    proc.load_data()
    proc.proc_mri()
    proc.do_beamformer()
    assert hasattr(proc, 'psds') 
    assert hasattr(proc, 'freqs')
    assert op.exists(proc.fnames.dics)
    
    # 'Reduce the epoch count to 3 for compute purposes'
    # tmp = []
    # for i in 0,1,2:
    #     tmp.append(next(proc.stcs))
    # proc.stcs = tmp
    # proc.do_label_psds()
    # assert hasattr(proc, 'label_ts')
    


