#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:41:59 2020

@author: stoutjd
"""

import pandas as pd
import numpy as np
import os, os.path as op
# os.chdir('/home/stoutjd/data/DEMO_ENIGMA/outputsBAK/enigma_outputs')

input_dir='/home/stoutjd/CAMCAN_ageValpha'
os.chdir(input_dir)

import glob


# file_list=glob.glob('????????_fs/Band_rel_power.csv')
# file_list=glob.glob('CompiledResults/????????_Band_rel_power.csv')
file_list=glob.glob('CompiledResults2/sub-CC*_Band_rel_power.csv')
curr_file=file_list[0]
dframe=pd.read_csv(curr_file, sep='\t')

#dframe['HASH_ID']= op.dirname(curr_file)[0:8]
dframe['HASH_ID']= op.basename(curr_file)[4:12]


for curr_file in file_list[1:]:
    print(curr_file)
    curr_dframe = pd.read_csv(curr_file, sep='\t')
    # curr_dframe['HASH_ID'] = op.dirname(curr_file)[0:8]
    curr_dframe['HASH_ID']= op.basename(curr_file)[4:12]
    dframe = dframe.append(curr_dframe)
    
    del curr_dframe

dframe = dframe.rename(columns={'Unnamed: 0':'Parcel'})

import seaborn as sns

plot_frame = dframe[['AlphaPeak','HASH_ID', 'Parcel']]
rois = ['pericalcarine-rh','postcentral-rh', 'bankssts-rh',
'pericalcarine-lh','postcentral-lh', 'bankssts-lh']

plot_frame=plot_frame[plot_frame.Parcel.isin(rois)]
plot_frame.dropna(inplace=True)

sns.histplot(data=plot_frame, x='Parcel', y='AlphaPeak')
sns.displot(plot_frame, x="AlphaPeak", kind="kde", bw_adjust=2, hue='Parcel')

import pylab
fig=pylab.Figure(figsize=(6,8), dpi=300)  #I dont think this is necessary
tmp=sns.displot(plot_frame, x="AlphaPeak", kind="kde", bw_adjust=2, hue='Parcel')
tmp.savefig(op.join(input_dir, 'SensoryHists.png'), dpi=300)



dframe.to_csv(op.join(input_dir, 'Compiled_bandrel.csv'), index=False)
