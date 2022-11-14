#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:04:07 2022

@author: jstout
"""
import os, os.path as op
from mne_bids import BIDSPath
import pandas as pd
import shutil

bids_root = '/data/EnigmaMeg/BIDS/Omega'
os.chdir(bids_root)

meg_dsets = glob.glob('sub-*/ses-*/meg/*.ds')
dframe=pd.DataFrame(meg_dsets, columns=['meg'])

def get_hs(dset):
    dirname = os.path.dirname(dset)
    hs_fname = glob.glob(op.join(dirname, '*headshape.pos'))
    if hs_fname==[]:
        return None
    else:
        return hs_fname[0]

dframe['hs_files']=dframe.meg.apply(get_hs)
# dframe['error']
dframe = dframe[dframe.hs_files.notnull()]

error=[]
for idx, row in dframe.iterrows():
    try:
        if op.exists(row['hs_files']):
            shutil.copy(row['hs_files'], row['meg'])
    except:
        error.append(row['meg'])
    



