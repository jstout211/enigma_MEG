#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:18:23 2023

@author: jstout
"""


import glob
import os, os.path as op
import mne 
import mne_bids
from mne_bids import BIDSPath, write_raw_bids, write_anat
import pandas as pd
import logging

logger = logging.Logger('base')
topdir='/fast/HCPPITT/sourcedata'
os.chdir(topdir)

dsets = glob.glob(op.join(topdir, 'HCP*','unprocessed','MEG','*','*.fif'))
subjids = [i.split('sourcedata')[-1].split('/')[1] for i in dsets]
dframe = pd.DataFrame(zip(dsets,subjids), columns=['fname','subjid'])

dframe['task']= dframe.fname.apply(get_dset_info, **{'return_type':'task'})
dframe['run']= dframe.fname.apply(get_dset_info, **{'return_type':'run'})
dframe['subjects_dir']=topdir + '/' +  dframe.subjid  + '/T1w'

def get_dset_info(fname, return_type=None):
    tasktype=op.basename(op.dirname(fname))
    if 'empty' in tasktype.lower():
        task='empty'
    elif 'close' in tasktype.lower():
        task='eyesclosed'
    elif 'open' in tasktype.lower():
        task='eyesopen'
    elif 'rest' in tasktype.lower():
        task='eyesNA'
    
    if tasktype[-1]=='1':
        run=1
    elif tasktype[-1]=='2':
        run=2
    else:
        run=1
    if return_type == 'run':
        return run
    else:
        return task


def do_auto_coreg(raw_fname, subject, subjects_dir):
    '''Localize coarse fiducials based on fsaverage coregistration
    and fine tuned with iterative headshape fit.'''
    raw_rest = mne.io.read_raw_fif(raw_fname)
    coreg = mne.coreg.Coregistration(raw_rest.info, 
                                     subject=subject,
                                     subjects_dir=subjects_dir, 
                                     fiducials='estimated')
    coreg.fit_fiducials(verbose=True)
    coreg.omit_head_shape_points(distance=5. / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=6, nasion_weight=.5, hsp_weight= 5, verbose=True)
    return coreg.trans
    
    
# =============================================================================
#  Make Transform from automated fit
# =============================================================================
failed=[]
for idx,row in dframe.iterrows():
    try:
        if (row.run==1) & (row.task not in ['empty','closed']):
            trans = do_auto_coreg(row.fname, row.subjid, row.subjects_dir)
            trans.save(op.join(op.dirname(op.dirname(row.fname)), f'{row.subjid}_trans.fif'))
    except:
        print(row.fname)
        failed.append(row.fname)


# =============================================================================
# MEG BIDS
# =============================================================================

bids_dir = op.join(topdir, 'BIDS')
for idx,row in dframe.iterrows():
    try:
        trans_fname = op.join(op.dirname(op.dirname(row.fname)), f'{row.subjid}_trans.fif')
        trans = mne.read_trans(trans_fname)
        raw = mne.io.read_raw_fif(row.fname)
        raw.info['line_freq'] = 60 
        if row.subjid[-2:]=='v2':
            ses='2'
        else:
            ses = '1'
        run = row.run
        run = str(run) 
        if len(run)==1: run='0'+run
        bids_path = BIDSPath(subject=row.subjid, session=ses, task=row.task,
                              run=run, root=bids_dir, suffix='meg')
        write_raw_bids(raw, bids_path, overwrite=True)
        logger.info(f'Successful MNE BIDS: {meg_fname} to {bids_path}')
    except BaseException as e:
        print(row.fname)
        logger.info(f'failed BIDS:  {str(e)}') 

# =============================================================================
# MRI BIDS
# =============================================================================

for idx,row in dframe.iterrows():
    if (row.task != 'eyesopen') & (row.run != 1):
        continue
    else:
        raw = mne.io.read_raw_fif(row.fname)
        trans_fname = op.join(op.dirname(op.dirname(row.fname)), f'{row.subjid}_trans.fif')
        trans = mne.read_trans(trans_fname)
        
        ses='1'
        t1w_bids_path = \
            BIDSPath(subject=row.subjid, session=ses, root=bids_dir, suffix='T1w')
    
        landmarks = mne_bids.get_anat_landmarks(
            image=op.join(row.subjects_dir, row.subjid, 'mri','T1.mgz'),
            info=raw.info,
            trans=trans,
            fs_subject=row.subjid,
            fs_subjects_dir=row.subjects_dir
            )
        
        # Write regular
        t1w_bids_path = write_anat(
            image=op.join(row.subjects_dir, row.subjid, 'mri','T1.mgz'),
            bids_path=t1w_bids_path,
            landmarks=landmarks,
            deface=False, 
            overwrite=True
            )

