#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:20:12 2020

@author: stoutjd
"""

import os
import numpy as np
from enigmeg.simulation.simulate_meg_data import generate_subjects_psuedomeg

import pytest

@pytest.mark.slow
@pytest.mark.sim
def test_generate_subjects_psuedomeg(tmpdir):
    from enigmeg.test_data.get_test_data import datasets
    
    #For elekta data
    elekta_dat = datasets().elekta
    
    subject = elekta_dat['subject']
    subjects_dir = elekta_dat['SUBJECTS_DIR']
    raw_fname = elekta_dat['meg_rest']
    eroom_fname = elekta_dat['meg_eroom']
    trans_fname = elekta_dat['trans']
    src_fname = elekta_dat['src']
    bem_fname = elekta_dat['bem']
    input_dir = elekta_dat['enigma_outputs']
    
    sfreq=100
    duration=10
    
    import os
    os.chdir(tmpdir)
    
    np.random.seed(31)
    generate_subjects_psuedomeg(subject, subjects_dir, raw_fname, trans_fname, bem_fname, 
                                src_fname, sfreq, duration, input_dir)

