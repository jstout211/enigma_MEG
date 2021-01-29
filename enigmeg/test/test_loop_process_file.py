#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:55:48 2021

@author: stoutjd
"""

import pandas as pd
import enigmeg
import mne
import os, os.path as op
import shutil

import pytest
from enigmeg.test_data import loop_test_data 
from enigmeg.test_data.get_test_data import datasets
from enigmeg.process_meg import main


def generate_short_test_data():
    '''This is unecessary to run - Data is stored in the git annex repo'''
    elekta_dat = datasets().elekta
    raw = mne.io.read_raw_fif(elekta_dat['meg_rest'])
    raw.crop(tmax=20)
    raw.resample(300)
    
    outfname = op.join(loop_test_data.__path__[0], 'short_elekta_rest.fif')
    raw.save(outfname) 
    
    eroom_raw = mne.io.read_raw_fif(elekta_dat['meg_eroom'])
    eroom_raw.crop(tmax=20)
    eroom_raw.resample(300)
    
    outfname_eroom = op.join(loop_test_data.__path__[0], 'short_elekta_eroom.fif')
    eroom_raw.save(outfname_eroom)
    
    # For CTF data it is necessary to use the CTF commandline tools
    tmp1 = 'newDs -time 1 21 -resample 4 ctf_rest.ds short_ctf_rest.ds'
    tmp2 = 'newDs -resample 4 ctf_eroom.ds short_ctf_eroom.ds'
    print('Must process CTF data manually\n{}\n{}'.format(tmp1, tmp2))

def parse_inputs(proc_file):
    # Load csv processing tab separated file
    proc_dframe = pd.read_csv(proc_file, sep='\t')    
    
    # Reject subjects with ignore flags
    keep_idx = proc_dframe.ignore.isna()   #May want to make a list of possible ignores
    proc_dframe = proc_dframe[keep_idx]
    
    for idx, dseries in proc_dframe.iterrows():
        print(dseries)
        
        dseries['output_dir']=op.expanduser(dseries['output_dir'])
        
        from types import SimpleNamespace
        info = SimpleNamespace()
        info.SUBJECTS_DIR = dseries['fs_subjects_dir']
        
        info.outfolder = op.join(dseries['output_dir'], dseries['subject'])
        info.bem_sol_filename = op.join(info.outfolder, 'bem_sol-sol.fif') 
        info.src_filename = op.join(info.outfolder, 'source_space-src.fif')
        

        os.environ['SUBJECTS_DIR']=dseries['fs_subjects_dir']
        
        #Determine if meg_file_path is a full path or relative path
        if not op.isabs(dseries['meg_file_path']):
            if op.isabs(dseries['meg_top_dir']):
                dseries['meg_file_path'] = op.join(dseries['meg_top_dir'], 
                                                   dseries['meg_file_path'])
            else:
                raise ValueError('This is not a valid path')
        
        #Perform the same check on the emptyroom data
        if not op.isabs(dseries['eroom_file_path']):
            if op.isabs(dseries['meg_top_dir']):
                dseries['eroom_file_path'] = op.join(dseries['meg_top_dir'], 
                                                   dseries['eroom_file_path'])
            else:
                raise ValueError('This is not a valid path')        
            
        inputs = {'filename' : dseries['meg_file_path'],
                  'subjid' : dseries['subject'],
                  'trans' : dseries['trans_file'],
                  'info' : info ,
                  'line_freq' : dseries['line_freq'],
                  'emptyroom_filename' : dseries['eroom_file_path']}
        main(**inputs)
        
def test_process_file(tmpdir):
    '''Generate a csv file and use this as input for the config file loop
    Loop over all entries in a tab separated csv file'''
    
    test_process_file = op.join(loop_test_data.__path__[0], 
                                'test_process_file.csv')
    
    test_dframe = pd.read_csv(test_process_file, delimiter='\t')
    
    output_dir = tmpdir #op.expanduser('~/Desktop/TEMP')
    
    # Process Elekta info
    elekta_dat = datasets().elekta    
    test_dframe.loc[2,'subject'] = elekta_dat['subject']
    test_dframe.loc[2, 'fs_subjects_dir'] = elekta_dat['SUBJECTS_DIR']
    test_dframe.loc[2, 'meg_top_dir'] = loop_test_data.__path__[0]
    test_dframe.loc[2, 'meg_file_path'] = 'short_elekta_rest.fif'
    test_dframe.loc[2, 'eroom_file_path'] = 'short_elekta_eroom.fif'
    test_dframe.loc[2, 'output_dir'] = output_dir
    test_dframe.loc[2, 'line_freq'] = 50
    test_dframe.loc[2, 'trans_file'] = elekta_dat['trans']
    enigma_subj_dir = op.join(output_dir, elekta_dat['subject'])
    if not op.exists(enigma_subj_dir):
                     os.mkdir(enigma_subj_dir)
    shutil.copy(elekta_dat['src'], enigma_subj_dir)
    shutil.copy(elekta_dat['bem'], enigma_subj_dir)
    
    # Process CTF info
    ctf_dat = datasets().ctf
    test_dframe.loc[3,'subject'] = ctf_dat['subject']
    test_dframe.loc[3, 'fs_subjects_dir'] = ctf_dat['SUBJECTS_DIR']
    test_dframe.loc[3, 'meg_top_dir'] = loop_test_data.__path__[0]
    test_dframe.loc[3, 'meg_file_path'] = 'short_ctf_rest.ds'
    test_dframe.loc[3, 'eroom_file_path'] = 'short_ctf_eroom.ds'
    test_dframe.loc[3, 'output_dir'] = output_dir
    test_dframe.loc[3, 'line_freq'] = 60        
    test_dframe.loc[3, 'trans_file'] = ctf_dat['trans']
    enigma_subj_dir = op.join(output_dir, ctf_dat['subject'])
    if not op.exists(enigma_subj_dir):
                     os.mkdir(enigma_subj_dir)
    shutil.copy(ctf_dat['src'], enigma_subj_dir)
    shutil.copy(ctf_dat['bem'], enigma_subj_dir)
    
    output_csv = op.join(output_dir, 'process.csv')  
    test_dframe.to_csv(output_csv, sep='\t', index=False)
    
    parse_inputs(output_csv)
    
    #Verify that the outputs have been created for the multiple inputs
    assert op.exists(op.join(output_dir, elekta_dat['subject'], 'Band_rel_power.csv'))
    assert op.exists(op.join(output_dir, ctf_dat['subject'], 'Band_rel_power.csv'))
    
    print(test_dframe)
    
    
    
    
