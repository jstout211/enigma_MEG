#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:52:35 2022

@author: stoutjd
"""

import os, os.path as op
import datalad.api as dl
import glob
from enigmeg import process_meg



download_path = os.path.expanduser('~')
openneuro_dset='ds004215'
downloads=\
    ['sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.json',
     'sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
     'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_channels.tsv',
     'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_coordsystem.json',
     'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_meg.ds',
     'sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-rest_run-01_meg.json']


def get_rest_data(dataset='ds004215',
                  download_location=download_path, 
                  downloads=downloads
                  ):
    '''Retrieve CTF rest data from NIMH HV dataset'''
    dl.install(
        path=op.join(download_path,dataset),
        source=f'https://github.com/OpenNeuroDatasets/{openneuro_dset}.git'
        )

    curr_dir = os.getcwd()
    os.chdir(op.join(download_path, dataset))    
    dl.get(path=downloads)
    
    os.chdir(curr_dir)    
    
get_rest_data(dataset='ds004215',
                  download_location=download_path, 
                  downloads=downloads
                  )

proc = process_meg.process(subject='ON02747',
                    bids_root=op.join(download_path, openneuro_dset),
                    session='01')

