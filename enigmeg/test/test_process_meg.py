#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:38:03 2020

@author: stoutjd
"""

# from hv_proc import test_config
from enigmeg.test_data.get_test_data import datasets
import pytest


filename = datasets().ctf['meg_rest']

# test_config.hariri['meg']

# from ..load_data import check_datatype, return_dataloader, load_data
from ..process_meg import check_datatype, return_dataloader, load_data

def test_check_datatype():
    assert check_datatype('test.fif')  == 'elekta'
    assert check_datatype('test.4d') == '4d'
    assert check_datatype('test.ds') == 'ctf'
    #assert ... KIT
    #assert ....
    
    #Verify that innapropriate inputs fail
    with pytest.raises(ValueError) as e:
        check_datatype('tmp.eeg')
    assert str(e.value) == 'Could not detect datatype'

def test_return_dataloader():
    import mne
    assert return_dataloader('ctf') == mne.io.read_raw_ctf
    assert return_dataloader('4d') == mne.io.read_raw_bti
    assert return_dataloader('elekta') == mne.io.read_raw_fif
    
def test_load_data():
    # from hv_proc import test_config
    filename = datasets().ctf['meg_rest'] #test_config.rest['meg']
    assert check_datatype(filename) == 'ctf'
    load_data(filename)     
    
