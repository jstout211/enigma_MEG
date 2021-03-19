#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:41:59 2020

@author: stoutjd
"""

import pandas as pd
import numpy as np
import os, os.path as op
import glob
import seaborn as sns
import pylab

def compile_group_outputs(enigma_outputs_path=None):
    '''Find all single subject enigma csv files and compile into a single
    dataframe'''

    os.chdir(enigma_outputs_path)
    filenames = glob.glob('*/Band_rel_power.csv')
    return filenames

def return_multisubject_dframe(filenames):
    '''Take a list of filenames 
    Load the csv filenames into a dataframe
    Stitch together single subject dataframes into one large dataframe'''
    
    list_of_dframes = list()
    for fname in filenames:
        print(fname)
        subj_dframe = pd.read_csv(fname, sep='\t')       
        subj_dframe['subject'] = fname.split('/')[0]
        list_of_dframes.append(subj_dframe)
        
    group_dframe = pd.concat(list_of_dframes)
    group_dframe = group_dframe.rename(columns={'Unnamed: 0':'Parcel'})
    return group_dframe
        
        
    


def test_plots(dframe):
    plot_frame = dframe[['AlphaPeak','subject', 'Parcel']]
    rois = ['pericalcarine_1-rh','postcentral_1-rh', 'bankssts_1-rh',
    'pericalcarine_1-lh','postcentral_1-lh', 'bankssts_1-lh']
    
    plot_frame=plot_frame[plot_frame.Parcel.isin(rois)]
    plot_frame.dropna(inplace=True)
    
    sns.histplot(data=plot_frame, x='Parcel', y='AlphaPeak')
    sns.displot(plot_frame, x="AlphaPeak", kind="kde", bw_adjust=2, hue='Parcel')
    

    # fig=pylab.Figure(figsize=(6,8), dpi=300)  #I dont think this is necessary
    # tmp=sns.displot(plot_frame, x="AlphaPeak", kind="kde", bw_adjust=2, hue='Parcel')
    # tmp.savefig(op.join(input_dir, 'SensoryHists.png'), dpi=300)

def test_compile_group_outputs():
    # from enigmeg import test_data    
    #Hack replace value w/ import parameter
    top_dir = '/data/test_data/GROUP/enigma_outputs'
    
    fnames = compile_group_outputs(top_dir)
    assert len(fnames) == 61

# if __name__ == '__main__':
    #dframe.to_csv(op.join(input_dir, 'Compiled_bandrel.csv'), index=False)
