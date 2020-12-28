# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, mne
from ..process_anatomical import anat_info, compile_fs_process_list
import enigma    
test_data_path = os.path.join(enigma.__path__[0], 'test_data')
os.environ['ENIGMA_REST_DIR'] = os.path.join(test_data_path, 'enigma_outputs')


def test_inputs():
    subjid='ctf_fs'
    subjects_dir = os.path.join(test_data_path, 'SUBJECTS_DIR')
    #subjects_dir=os.path.join(os.environ['HOME'],'hv_proc/MRI')
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
    assert compile_fs_process_list(info) == ['recon-all -autorecon3 -s ctf_fs']
    info=test_inputs() 
    info.run_unprocessed = True 
    info.fs_surf_contents.remove('lh.pial')
    assert compile_fs_process_list(info) == ['recon-all -autorecon2 -s ctf_fs',
                                             'recon-all -autorecon3 -s ctf_fs']
    info=test_inputs()
    info.run_unprocessed = True
    info.fs_mri_contents.remove('brainmask.mgz')
    #Process All
    assert compile_fs_process_list(info) == ['recon-all -autorecon1 -s ctf_fs',
                                             'recon-all -autorecon2 -s ctf_fs',
                                             'recon-all -autorecon3 -s ctf_fs']
    info=test_inputs()
    info.recon1=True
    assert compile_fs_process_list(info) == ['recon-all -autorecon1 -s ctf_fs']

    info=test_inputs()
    info.recon2=True   
    assert compile_fs_process_list(info) == ['recon-all -autorecon2 -s ctf_fs']
    
    info=test_inputs()
    info.recon3=True   
    assert compile_fs_process_list(info) == ['recon-all -autorecon3 -s ctf_fs']
    
   
    
    
