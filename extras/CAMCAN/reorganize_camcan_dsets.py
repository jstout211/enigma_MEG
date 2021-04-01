#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:45:56 2020

@author: stoutjd
"""

import pandas as pd
import mne 
import os, os.path as op
import shutil
import glob

#Configuration Components
camcan_topdir = '/data/EnigmaMeg/CAMCAN/camcan/cc700'
camcan_anatdir = op.join(camcan_topdir,
                         'mri/pipeline/release004/BIDS_20190411/anat')
camcan_megdir = op.join(camcan_topdir,
                        'meg/pipeline/release004/BIDS_20190411')
camcan_freesurfer_dir = ''

# camcan_new_bidsout = '/data/EnigmaMeg/CAMCAN/camcan_mne_bids'
camcan_new_bidsout = '/data/EnigmaMeg/CAMCAN_testset/camcan'

#Define the different MEG datasets
tmp = glob.glob(camcan_megdir+'/*')
meg_types = [op.basename(i) for i in tmp]

def return_subject_anatdir(subjid=None, topdir=None):
    return op.join(topdir, subjid, 'anat')    

def return_subject_megdir(subjid=None, topdir=None, meg_task=None):
    task_id=meg_task.split('_')[1]
    return op.join(topdir,meg_task,subjid,'ses-'+task_id)   

def return_outputdir_anat(subjid=None, topdir=None):
    return op.join(camcan_new_bidsout, subjid)

## Make Subject Directory
# Sylink Anatomy Folder
# Make Meg folder
#   <br>Iterate over all meg types and meg files
#   <br>Symlink all task/rest/emptyroom datasets into meg folder

def create_links(dseries=None):
    #Make subject directory
    if not os.path.exists(dseries['output_subject_dir']):
        os.mkdir(dseries['output_subject_dir'])
    
    #Make anatomy symlinks
    if not os.path.exists(dseries['output_anat_dir']):
        os.symlink(dseries['input_anat_dir'], dseries['output_anat_dir'])
        
    #Make meg directory
    if not os.path.exists(dseries['output_meg_dir']):
        os.mkdir(dseries['output_meg_dir'])
        
    #Loop over meg tasks and create symlinks for all files into the folder:
    for task_id in ['meg_rest_mf', 'meg_emptyroom', 'meg_rest_raw']:#, 'meg_smt_mf']:
#     for task_id in meg_types:
        print(task_id)
        #Skip - the no movement correction data
        if 'nomovecomp' in task_id:
            continue
        if task_id == 'meg_emptyroom':
            try:
                tmp = op.dirname(dseries['input_'+task_id])
                eroom = glob.glob(op.join(tmp, 'emptyroom','*room*.fif'))[0]
                dset_basename=op.basename(eroom)
                output_dset_path = op.join(dseries['output_meg_dir'], dset_basename)
                if not os.path.exists(eroom):
                    continue
                if not os.path.exists(output_dset_path):
                    print(eroom, output_dset_path)
                    os.symlink(eroom, output_dset_path)    
            except:
                print('Error linking {}:'.format(dseries['input_'+task_id]))
            
        if task_id == 'meg_rest_mf':
            for input_dset_path in glob.glob(dseries['input_'+task_id]+'/meg/*'):
                dset_basename=op.basename(input_dset_path)
                output_dset_path = op.join(dseries['output_meg_dir'], dset_basename)
                if not os.path.exists(input_dset_path):
                    continue
                if not os.path.exists(output_dset_path):
                    print(input_dset_path, output_dset_path)
                    #shutil.copyfile(input_dset_path, output_dset_path)
                    os.symlink(input_dset_path, output_dset_path)
        
        if task_id == 'meg_rest_raw':
            for input_dset_path in glob.glob(dseries['input_'+task_id]+'/meg/*'):
                dset_basename=op.basename(input_dset_path)
                output_dset_path = op.join(dseries['output_meg_dir'], dset_basename)
                output_dset_path = output_dset_path.replace('.fif', '_raw.fif')
                if not os.path.exists(input_dset_path):
                    continue
                if not os.path.exists(output_dset_path):
                    print(input_dset_path, output_dset_path)
                    #shutil.copyfile(input_dset_path, output_dset_path)
                    os.symlink(input_dset_path, output_dset_path)
            
        

def link_anat_folder(dframe=None, subjid=None):
    idx=dframe[dframe['subject']==subjid]
    os.symlink(dframe.loc[idx,], dframe['output_subjectdir'])
    
#Get the subject ids listed in the camcan download and create a dataframe from these IDs
tmp = glob.glob(camcan_anatdir+'/sub-*')
subjids = [op.basename(i) for i in tmp]
camcan_dframe=pd.DataFrame(subjids, columns=['subject'])

#Get anatomy and MEG directories
camcan_dframe['input_anat_dir']=camcan_dframe.subject.apply(return_subject_anatdir, 
                                                      topdir=camcan_anatdir)
for megid in meg_types:
    camcan_dframe['input_'+megid]=camcan_dframe.subject.apply(return_subject_megdir,
                                                     topdir=camcan_megdir,
                                                     meg_task=megid)

#Define the output folders
camcan_dframe['output_subject_dir']=camcan_dframe.subject.apply(return_outputdir_anat, 
                                                       topdir=camcan_new_bidsout)
camcan_dframe['output_anat_dir']=camcan_dframe['output_subject_dir']+'/anat'
camcan_dframe['output_meg_dir']=camcan_dframe['output_subject_dir']+'/meg'

#Loop over data to create the symlinks
for i in camcan_dframe.index:
    create_links(camcan_dframe.loc[i])
