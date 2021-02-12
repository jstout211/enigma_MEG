#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:38:03 2020

@author: stoutjd
"""

# from hv_proc import test_config
from enigmeg.test_data.get_test_data import datasets
from enigmeg.process_meg import main 
import pytest
import os
import os.path as op
from numpy import allclose
import pandas as pd
from types import SimpleNamespace 


from ..process_meg import check_datatype, return_dataloader, load_data
from ..process_meg import make_report

def test_check_datatype():
    assert check_datatype('test.fif')  == 'elekta'
    assert check_datatype('test.4d') == '4d'
    assert check_datatype('test.ds') == 'ctf'
    assert check_datatype('test.sqd') == 'kit'
    
    #Verify that innapropriate inputs fail
    with pytest.raises(ValueError) as e:
        check_datatype('tmp.eeg')
    assert str(e.value) == 'Could not detect datatype'

def test_return_dataloader():
    import mne
    assert return_dataloader('ctf') == mne.io.read_raw_ctf
    assert return_dataloader('4d') == mne.io.read_raw_bti
    assert return_dataloader('elekta') == mne.io.read_raw_fif
    assert return_dataloader('kit') == mne.io.read_raw_kit
    
def test_load_data():
    filename = datasets().ctf['meg_rest'] 
    assert check_datatype(filename) == 'ctf'
    load_data(filename)  
    
# def test_make_report(tmpdir):
#     'Test that the report function generates an html file'
#     dat = datasets().ctf
#     make_report(dat['subject'],
#                 dat['SUBJECTS_DIR'],
#                 None,
#                 tmpdir)
#     assert os.path.exists(tmpdir.join('QA_report.html'))

@pytest.mark.meg    
@pytest.mark.slow
def test_main_ctf(tmpdir):
    '''
    CTF - Specific Test - Using Openneuro data from NIH MEG Core
    
    Perform the full process on the MEG data and compare the output
    spectra and binned power estimates'''
    
    
    #Get ctf data paths from git annex repo
    test_dat = datasets().ctf
    inputs=SimpleNamespace(**test_dat)
    
    info=SimpleNamespace()
    info.bem_sol_filename = bem=inputs.bem  
    info.src_filename = inputs.src
    info.outfolder = tmpdir  #Override the typical enigma_outputs folder
    
    os.environ['SUBJECTS_DIR']=inputs.SUBJECTS_DIR
    
    main(filename=inputs.meg_rest,
         subjid=inputs.subject,
         trans=inputs.trans,
         emptyroom_filename=inputs.meg_eroom,
         info=info,
         line_freq=60
         )
    
    standard_csv_path = op.join(test_dat['enigma_outputs'], 'ctf_fs', 
                                'Band_rel_power.csv')
    standard_dframe = pd.read_csv(standard_csv_path, delimiter='\t')
    test_dframe = pd.read_csv(tmpdir.join('Band_rel_power.csv'), delimiter='\t')
    
    allclose(standard_dframe.iloc[:,1:], test_dframe.iloc[:,1:])

@pytest.mark.meg    
@pytest.mark.slow
def test_main_elekta(tmpdir):
    '''
    MEGIN/Elekta - Specific Test - Using fif data from CAMCAN 
    
    Perform the full process on the MEG data and compare the output
    spectra and binned power estimates'''
    
    
    #Get elekta data paths from git annex repo
    test_dat = datasets().elekta
    inputs=SimpleNamespace(**test_dat)
    
    info=SimpleNamespace()
    info.bem_sol_filename = inputs.bem  
    info.src_filename = inputs.src
    info.outfolder = tmpdir  #Override the typical enigma_outputs folder
    
    os.environ['SUBJECTS_DIR']=inputs.SUBJECTS_DIR
    
    main(filename=inputs.meg_rest,
         subjid=inputs.subject,
         trans=inputs.trans,
         emptyroom_filename=inputs.meg_eroom,
         info=info,
         line_freq=50  #CAMCAN mains freq
         )
    
    standard_csv_path = op.join(test_dat['enigma_outputs'], 'elekta_fs', 
                                'Band_rel_power.csv')
    standard_dframe = pd.read_csv(standard_csv_path, delimiter='\t')
    test_dframe = pd.read_csv(tmpdir.join('Band_rel_power.csv'), delimiter='\t')
    
    allclose(standard_dframe.iloc[:,1:], test_dframe.iloc[:,1:])
    
    

# def test_QA_plot(tmpdir):
#     import bunch
#     info = bunch.Bunch()
#     meg_filename = 
#     subjid = 
#     subjects_dir = 
#     raw = mne.io.read_raw_ctf(meg_filename)
#     trans = mne.read_trans()
#     info.subjid, info.subjects_dir = subjid, subjects_dir
#     info.outfolder = tmpdir
#     plot_QA_head_sensor_align(info, raw, trans ) 
    
    
    
    
    
    
    
    
    
    
    
