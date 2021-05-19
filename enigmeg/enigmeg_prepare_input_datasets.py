#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:01:15 2021

@author: jstout

TODO:
    anonymization
    sort subject data into session/runs
    select duplicates
        REMOVE SECTION drop_duplicates !
    add subject index to dataframe


based on code from:
    https://mne.tools/mne-bids/stable/auto_examples/convert_group_studies.html#sphx-glr-auto-examples-convert-group-studies-py
    # Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
    #          Teon Brooks <teon.brooks@gmail.com>
    #          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
    #
    # License: BSD (3-clause)

"""

# =============================================================================
# Import Functions
# =============================================================================
import os, os.path as op
import shutil

import mne
from mne.datasets import eegbci

from mne_bids import (write_raw_bids, BIDSPath,
                      get_anonymization_daysback, make_report,
                      print_dir_tree)
from mne_bids.stats import count_events

import pandas as pd
import glob

# =============================================================================
# Convenience Functions
# =============================================================================
def dir_at_pos(file_path, position=-1):
    '''Split directories and return the one at position'''
    return os.path.normpath(file_path).split(os.sep)[position]

def subjid_from_filename(filename, position=[0], split_on='_', multi_index_cat='_'):
    tmp = os.path.splitext(filename)
    if len(tmp)>2:
        return 'Error on split extension - Possibly mutiple "." in filename'
    if not isinstance(position, list):
        print('The position variable must be a list even if a single entry')
        raise ValueError
    filename = os.path.splitext(filename)[0]
    filename_parts = filename.split(split_on)    
    subjid_components = [filename_parts[idx] for idx in position]
    if isinstance(subjid_components, str) or len(subjid_components)==1:
        return subjid_components[0]
    elif len(subjid_components)>1:
        return multi_index_cat.join(subjid_components)
    else:
        return 'Error'

def print_multiple_mris(mri_dframe):
    subjs_mri_count = mri_dframe.mri_subjid.value_counts()
    subjs_wMulti_mri = subjs_mri_count[subjs_mri_count>1]
    mri_dframe[mri_dframe.mri_subjid.isin(subjs_wMulti_mri.index)]

# =============================================================================
# CONFIG SECTION - To be separated into new file
# =============================================================================
line_freq = 60.0

mri_subjid_on_filename = False
mri_subjid_on_directory = True
mri_subjid_on_directory_level = -2
mri_top_level_dir = '/fast/BIDS/prep_files/MRI'
if mri_top_level_dir[-1]==os.sep: mri_top_level_dir=mri_top_level_dir[:-1]

meg_subjid_on_filename = True
meg_subjid_on_filename_split_val = '_'
meg_subjid_on_filename_split_level = 0
meg_top_level_dir = '/fast/BIDS/prep_files/rest_data'
if meg_top_level_dir[-1]==os.sep: meg_top_level_dir=meg_top_level_dir[:-1]

#Initiate MRI dataframe
mri_glob =f'{mri_top_level_dir}/*/*.nii'
mri_inputs = glob.glob(mri_glob)
mri_dframe = pd.DataFrame(mri_inputs, columns=['full_mri_path'])
mri_dframe['mri_subjid'] =  mri_dframe.full_mri_path.apply(dir_at_pos, position=-2)

#Initiate MEG dataframe
meg_glob = op.join(meg_top_level_dir,'*.ds') 
meg_inputs = glob.glob(meg_glob)
meg_dframe = pd.DataFrame(meg_inputs, columns=['full_meg_path'])
meg_dframe['meg_filename'] = meg_dframe.full_meg_path.apply(dir_at_pos, position=-1)
meg_dframe['meg_subjid'] = meg_dframe.meg_filename.apply(subjid_from_filename, position=[0])
# =============================================================================
# END CONFIG
# =============================================================================

print(f'Searching for MRIs in: {mri_top_level_dir}')
print(f'Searching for MEG files in: {meg_top_level_dir}')
print(f'Found MRI files (possibly more than one per subject: {len(mri_dframe)}')
print(f'Found meg files (possibly more than one per subject): {len(meg_dframe)}')

combined_dframe  = pd.merge(mri_dframe, meg_dframe, left_on='mri_subjid', 
                            right_on='meg_subjid')

#!!! For this set only select only the first MEG set
combined_dframe = combined_dframe.drop_duplicates(subset='meg_subjid')


bids_dir = './bids_out'
if not os.path.exists(bids_dir): os.mkdir(bids_dir)

# =============================================================================
# Convert MEG
# =============================================================================
for idx, row in combined_dframe.iterrows():
    print(idx)
    print(row)
    subject = row.meg_subjid
    mri_fname = row.full_mri_path
    raw_fname = row.full_meg_path
    output_path = bids_dir
    
    raw = mne.io.read_raw_ctf(raw_fname)  #Change this - should be generic for meg vender
    raw.info['line_freq'] = line_freq 

    sub = "{0:0=4d}".format(idx)
    ses = '01'
    task = 'rest'
    run = '01'
    bids_path = BIDSPath(subject=sub, session=ses, task=task,
                         run=run, root=output_path)
    
    write_raw_bids(raw, bids_path)   

# =============================================================================
# Convert MRI - to NON-Anonymous!!!!
# =============================================================================
for idx, row in combined_dframe.iterrows():
    print(idx)
    print(row)
    subject = row.meg_subjid
    mri_fname = row.full_mri_path
    raw_fname = row.full_meg_path
    output_path = bids_dir
    
    raw = mne.io.read_raw_ctf(raw_fname)  #Change this - should be generic for meg vender
    raw.info['line_freq'] = line_freq 

    sub = "{0:0=4d}".format(idx)
    ses = '01'
    task = 'rest'
    run = '01'
    bids_path = BIDSPath(subject=sub, session=ses, task=task,
                         run=run, root=output_path)
    
    write_raw_bids(raw, bids_path)  


    
report = make_report(bids_dir)
# def check_inputs(bids_root=None, line_freq=None, meg_data_dir=None, 
#                  subjects_dir=None, subjids='all'):
#     '''Glob all impotant directories and determine the missing inputs'''
    
        
#     if subjids=='all':
#         print('''Parsing all meg studies can take a long time depending on the 
#               number of datasets in the search''')
#         all_meg_datasets = glob.glob(meg_glob)
#         all_mri_datasets = glob.glob(mri_glob)
    



