#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:08:12 2020

@author: stoutjd
"""
import os
import mne

class anat_info():
    '''Collect information for processing the data'''
    def __init__(self, **kwargs):
        self.recon1=False
        self.recon2=False
        self.recon3=False
        self.setup_source=False
        self.run_unprocessed=False
        self.subjid=kwargs['subjid']
        if 'SUBJECTS_DIR' in kwargs:
            self.subjects_dir=kwargs['SUBJECTS_DIR']
        else:
            self.subjects_dir=os.environ['SUBJECTS_DIR']
        if self.subjid not in os.listdir(self.subjects_dir):
            raise ValueError('''{} not in {}.  If unexpected: 
                1) check that the subject is in the SUBJECTS_DIR, or 
                2) Set subjects_dir at the commandline'''.format(self.subjid, self.subjects_dir))
        self.fs_subj_dir=os.path.join(self.subjects_dir, self.subjid)
        self.fs_mri_contents=os.listdir(os.path.join(self.fs_subj_dir, 'mri')) 
        self.fs_surf_contents=os.listdir(os.path.join(self.fs_subj_dir, 'surf'))
        self.fs_label_contents=os.listdir(os.path.join(self.fs_subj_dir, 'label'))

def compile_fs_process_list(info):
    '''Verifies necessary steps for processing and returns a list'''
    process_steps=[]
    proc_downstream=0
    if info.run_unprocessed:
        if ('brainmask.mgz' not in info.fs_mri_contents) | info.recon1:
            process_steps.append('recon-all -autorecon1 -s {}'.format(info.subjid))
            proc_downstream = True
        if ('lh.pial' not in info.fs_surf_contents) | ('rh.pial' not in info.fs_surf_contents) | info.recon2 | proc_downstream:
            process_steps.append('recon-all -autorecon2 -s {}'.format(info.subjid))
            proc_downstream = True
        if ('lh.aparc.annot' not in info.fs_label_contents) | ('rh.aparc.annot' not in info.fs_label_contents) | info.recon3 | proc_downstream:
            process_steps.append('recon-all -autorecon3 -s {}'.format(info.subjid))
            proc_downstream = True
            
    # If run_unprocessed is not set.  All independent steps must best to run manually
    if info.recon1:
        process_steps.append('recon-all -autorecon1 -s {}'.format(info.subjid))
    if info.recon2:
        process_steps.append('recon-all -autorecon2 -s {}'.format(info.subjid))
    if info.recon3:
        process_steps.append('recon-all -autorecon3 -s {}'.format(info.subjid))
    return process_steps        


# def compile_process_list(info):
#     '''Main function for processing freesurfer and source level setup'''
#     if ('brainmask.mgz' not in info.fs_mri_contents:
#         print('Brainmask not present: Adding autorecon1 to process list')
#     if ('lh.pial' not in info.fs_surf_contents) | ('rh.pial' not in info.fs_surf_contents):
#         print('Left or Right hemi not present: Running autorecon2')
#     if ('lh.aparc.annot' not in info.fs_label_contents) | ('rh.aparc.annot' not in info.fs_label_contents):
#         print('Labels not present: Running autorecon3')
#     if ('lh.mid' not in info.fs_surf_contents) | ('rh.mid' not in info.fs_surf_contents):
#         print('Running mne.setup_source_space()')        
#         from mne import setup_source_space
#         setup_source_space(subject=info.subjid, subjects_dir=subjects_dir, n_jobs=1)   
        
    
if __name__=='__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjects_dir', help='''Freesurfer subjects_dir can be 
                        assigned at the commandline if not already exported.''')
    parser.add_argument('-subjid', help='''Define subjects id (folder name)
                        in the SUBJECTS_DIR''')
    parser.add_argument('-recon_check', help='''Process all anatomical steps that
                        have not been completed already.  This will check the major
                        outputs from autorecon1, 2, 3, and mne source setup and
                        proceed with the processing. The default is set to TRUE''')
    parser.add_argument('-recon1', help='''Force recon1 to be processed''', action='store_true')
    parser.add_argument('-recon2', help='''Force recon2 to be processed''', action='store_true')
    parser.add_argument('-recon3', help='''Force recon3 to be processed''', action='store_true')
    parser.add_argument('-setup_source', help='''Runs the setup source space processing
                        in mne python to create the BEM model''', action='store_true')
    parser.add_argument('-run_unprocessed', help='''Checks for all unrun processes and
                        runs any additional steps for inputs to the source model''')
    parser.description='''Processing for the anatomical inputs of the enigma pipeline'''
    args = parser.parse_args()
    if not args.subjid: raise ValueError('Subject ID must be set')
    
    #Initialize Defaults
    info=anat_info(subjid=args.subjid, SUBJECTS_DIR=args.subjects_dir)
    #Override defaults with commandline options
    if args.recon1: info.recon1=True
    if args.recon2: info.recon2=True
    if args.recon3: info.recon3=True
    if args.setup_source: info.setup_source=True
    if args.run_unprocessed: info.run_unprocessed=True
    

    

def test_inputs():
    subjid='APBWVFAR_fs'
    subjects_dir=os.path.join(os.environ['HOME'],'hv_proc/MRI')
    info=anat_info(subjid=subjid, SUBJECTS_DIR=subjects_dir)
    assert info.subjid==subjid
    assert info.subjects_dir==subjects_dir
    assert info.recon1==False
    assert info.recon2==False
    assert info.recon3==False
    assert info.setup_source==False
    return info
    

def test_compile_fs_process_list():
    info=test_inputs()
    info.run_unprocessed = True
    #Full list
    assert compile_fs_process_list(info) == []
    info.fs_label_contents.remove('lh.aparc.annot')
    info.fs_label_contents.remove('rh.aparc.annot')
    assert compile_fs_process_list(info) == ['recon-all -autorecon3 -s APBWVFAR_fs']
    info=test_inputs() 
    info.run_unprocessed = True 
    info.fs_surf_contents.remove('lh.pial')
    assert compile_fs_process_list(info) == ['recon-all -autorecon2 -s APBWVFAR_fs',
                                             'recon-all -autorecon3 -s APBWVFAR_fs']
    info=test_inputs()
    info.run_unprocessed = True
    info.fs_mri_contents.remove('brainmask.mgz')
    #Process All
    assert compile_fs_process_list(info) == ['recon-all -autorecon1 -s APBWVFAR_fs',
                                             'recon-all -autorecon2 -s APBWVFAR_fs',
                                             'recon-all -autorecon3 -s APBWVFAR_fs']
    info=test_inputs()
    info.recon1=True
    assert compile_fs_process_list(info) == ['recon-all -autorecon1 -s APBWVFAR_fs']

    info=test_inputs()
    info.recon2=True   
    assert compile_fs_process_list(info) == ['recon-all -autorecon2 -s APBWVFAR_fs']
    
    info=test_inputs()
    info.recon3=True   
    assert compile_fs_process_list(info) == ['recon-all -autorecon3 -s APBWVFAR_fs']
    
   
    
    
    
    


