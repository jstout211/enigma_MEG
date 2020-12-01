#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:52:16 2020

@author: stoutjd
"""

#get the trans function
import nibabel as nib
import os, os.path as op
import numpy as np
import glob
import json
from mne.transforms import apply_trans
import mne
from mne_bids.read import fit_matched_points
from mne_bids.read import _extract_landmarks

np.set_printoptions(suppress=True)


def trans_from_json(t1w_path=None, coordsys_json_path=None,
                    meg_path=None, subjid=None, plot_coreg=False,
                    subjects_dir=None):
    '''
    Script to extract the transformation matrix from the CAMCAN datasets
    
    Data must be reorganized before implementing
        The script expects root_dir/subjid/{meg,anat}/....
        Use the file - .reorganize_camcan_dsets.py to organize data
    '''
    
    
    
    ##### Code below is from the mne_bids package
    # '0.6.dev0'
    # commit 18cc4f912495e8de8ab30d9c3077b19c923bebf4
    # Specific function is mne_bids.get_head_mri_trans
    # Modifications were made to bypass some of the bids requirements
    # Also the LPA,RPA,Nasion are pulled from the coordsys meg file not T1.json
    # >>>
    
    # Get MRI landmarks from MEGcoordsys JSON sidecar
    with open(coordsys_json_path, 'r', encoding='utf-8-sig') as f:
        coordsys_json = json.load(f)
    mri_coords_dict = coordsys_json.get('AnatomicalLandmarkCoordinates', dict())
    mri_landmarks = np.asarray((mri_coords_dict.get('LPA', np.nan),
                                mri_coords_dict.get('NAS', np.nan),
                                mri_coords_dict.get('RPA', np.nan)))    
    
    if np.isnan(mri_landmarks).any():
        raise RuntimeError('Could not parse T1w sidecar file: "{}"\n\n'
                           'The sidecar file MUST contain a key '
                           '"AnatomicalLandmarkCoordinates" pointing to a '
                           'dict with keys "LPA", "NAS", "RPA". '
                           'Yet, the following structure was found:\n\n"{}"'
                           .format(coordsys_json_path, coordsys_json))
    
    # The MRI landmarks are in "voxels". We need to convert the to the
    # neuromag RAS coordinate system in order to compare the with MEG landmarks
    # see also: `mne_bids.write.write_anat`
    # t1w_path = t1w_json_path.replace('.json', '.nii')
    if not op.exists(t1w_path):
        t1w_path += '.gz'  # perhaps it is .nii.gz? ... else raise an error
    if not op.exists(t1w_path):
        raise RuntimeError('Could not find the T1 weighted MRI \
                           Tried: "{}" but it does not exist.'
                           .format(t1w_path))
    
    mri_landmarks = mri_landmarks * 1e-3
    
    raw =  mne.io.read_raw_fif(meg_path)
    meg_coords_dict = _extract_landmarks(raw.info['dig'])
    meg_landmarks = np.asarray((meg_coords_dict['LPA'],
                                meg_coords_dict['NAS'],
                                meg_coords_dict['RPA']))
    
    # Given the two sets of points, fit the transform
    trans_fitted = fit_matched_points(src_pts=meg_landmarks,
                                      tgt_pts=mri_landmarks,
                                      weights=[1,10,1])
    trans = mne.transforms.Transform(fro='head', to='mri', trans=trans_fitted)
    
    # <<< End of block
    
    if subjects_dir==None:
        if 'SUBJECTS_DIR' not in os.environ:
            print('SUBJECTS_DIR must either be an environmental variable\
                  or declared on the commandline.')
            raise(ValueError)
        else:
            subjects_dir=os.environ['SUBJECTS_DIR']
        
    pial_path=op.join(subjects_dir, '{}/surf/lh.pial'.format(subjid))
    _,_,tmp = nib.freesurfer.io.read_geometry(pial_path, read_metadata=True)
    offset = tmp['cras']
    
    trans['trans'][0,3]-=(offset[0]*.001)  
    trans['trans'][1,3]-=(offset[1]*.001)
    trans['trans'][2,3]-=(offset[2]*.001)
    
    trans_filename=os.path.join(os.path.dirname(t1w_path),
                                subjid+'-trans.fif')
    
    trans.save(trans_filename)
    print(trans)
    
    if plot_coreg==True:
        mne.viz.plot_alignment(raw.info, dig=True,
                               subjects_dir='./SUBJECTS_DIR', 
                               subject=subjid,
                               trans=trans,
                               surfaces=['pial','head'],
                               meg=False)#True)
        _ = input('Enter anything to continue')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-camcan_dir', help='Top directory of CAMCAN converted format')
    parser.add_argument('-subjid', help='Subjid of dataset')
    parser.add_argument('-plot_coreg', help='Bring up a pyvista image with coreg',
                        action='store_true')
    parser.add_argument('-subjects_dir', help='Overrides SUBJECTS_DIR set in environment')
    args=parser.parse_args()
    
    if not args.camcan_dir or not args.subjid:
        print('The camcan_dir and subjid options must be supplied')
        exit()
        
    os.chdir(args.camcan_dir)
    
    subjid=args.subjid
    meg_path='{}/meg/{}_ses-rest_task-rest_proc-sss.fif'.format(subjid,subjid)
    coordsys_path = '{}/meg/{}_ses-rest_task-rest_proc-sss_coordsystem.json'.format(subjid,subjid)
    
    print(meg_path)
    tmp_path=os.path.join(args.camcan_dir, subjid, 'anat')
    print(tmp_path)
    print(glob.glob(tmp_path+'/*T1w*.nii.gz'))
    t1w_path = glob.glob(tmp_path+'/*T1w*.nii.gz')[0]
    
    if args.subjects_dir:
        subjects_dir=args.subjects_dir
    else:
        subjects_dir=None
    
    print(t1w_path)
    trans_from_json(t1w_path=t1w_path, coordsys_json_path=coordsys_path,
                    meg_path=meg_path, subjid=subjid, plot_coreg=args.plot_coreg,
                    subjects_dir=subjects_dir)
    
    

    
