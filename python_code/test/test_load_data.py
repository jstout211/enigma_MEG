#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:38:03 2020

@author: stoutjd
"""

from hv_proc import test_config
filename = test_config.hariri['meg']

def test_check_datatype():
    from ..load_data import check_datatype
    assert check_datatype('test.fif')  == 'elekta'
    assert check_datatype('test.4d') == '4d'
    assert check_datatype('test.ds') == 'ctf'
    #assert ... KIT
    #assert ....

def test_return_dataloader():
    from ..load_data import return_dataloader
    import mne
    assert return_dataloader('ctf') == mne.io.read_raw_ctf
    assert return_dataloader('4d') == mne.io.read_raw_bti
    assert return_dataloader('elekta') == mne.io.read_raw_fif
    
