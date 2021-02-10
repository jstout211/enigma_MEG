# -*- coding: utf-8 -*-
"""

"""

import os, mne
from enigmeg.process_anatomical import anat_info, compile_fs_process_list
import enigmeg  
from enigmeg.test_data.get_test_data import datasets 
#from enigmeg.process_anatomical import main
import pytest 
from numpy import allclose
from types import SimpleNamespace 

test_data_path = os.path.join(enigmeg.__path__[0], 'test_data')
os.environ['ENIGMA_REST_DIR'] = os.path.join(test_data_path, 'enigma_outputs')


def test_inputs():
    subjid='ctf_fs'
    subjects_dir = os.path.join(test_data_path, 'SUBJECTS_DIR')
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

#######################
# This test needs to incorportate commandline call through subprocess
# or moving the freesurfer/MNE anatomical processing to a function
########################
    
# @pytest.mark.slow
# def test_main_anatomical(tmpdir):
#     '''
#     '''
    
    
#     #Get ctf data paths from git annex repo
#     test_dat = datasets().ctf
#     inputs=SimpleNamespace(**test_dat)
    
#     info=SimpleNamespace()
#     info.bem_sol_filename = bem=inputs.bem  
#     info.src_filename = inputs.src
#     info.outfolder = tmpdir  #Override the typical enigma_outputs folder
    
#     os.environ['SUBJECTS_DIR']=inputs.SUBJECTS_DIR
    
#     import subprocess
    
#     main(filename=inputs.meg_rest,
#          subjid=inputs.subject,
#          trans=inputs.trans,
#          emptyroom_filename=inputs.meg_eroom,
#          info=info,
#          line_freq=60
#          )
    
#     standard_csv_path = op.join(test_dat['enigma_outputs'], 'ctf_fs', 
#                                 'Band_rel_power.csv')
#     standard_dframe = pd.read_csv(standard_csv_path, delimiter='\t')
#     test_dframe = pd.read_csv(tmpdir.join('Band_rel_power.csv'), delimiter='\t')
    
#     allclose(standard_dframe.iloc[:,1:], test_dframe.iloc[:,1:])
    
    
    
