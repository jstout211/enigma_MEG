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

os.chdir('/home/stoutjd/data/CAMCAN')
# fs_aff = nib.load('SUBJECTS_DIR/sub-CC121106/mri/T1.mgz').affine
# t1_aff = nib.load('sub-CC121106/anat/sub-CC121106_T1w.nii.gz').affine

root_dir = '/home/stoutjd/data/CAMCAN'
subjid = 'sub-CC121106'
os.chdir(root_dir)

#may need to fix filenames on CAMCAN bids - mne_bids throws warnings
meg_path='{}/meg/{}_ses-rest_task-rest_proc.fif'.format(subjid,subjid)


tmp_path=os.path.join(root_dir, subjid, 'anat')
t1w_json_path = glob.glob(tmp_path+'/*T1w*.json')[0]



##### Code below is from the mne_bids package
# '0.6.dev0'
# commit 18cc4f912495e8de8ab30d9c3077b19c923bebf4
# Specific function is mne_bids.get_head_mri_trans
# Modifications were made to bypass some of the bids requirements

# Get MRI landmarks from the JSON sidecar
with open(t1w_json_path, 'r', encoding='utf-8-sig') as f:
    t1w_json = json.load(f)
mri_coords_dict = t1w_json.get('AnatomicalLandmarkCoordinates', dict())
mri_landmarks = np.asarray((mri_coords_dict.get('LPA', np.nan),
                            mri_coords_dict.get('NAS', np.nan),
                            mri_coords_dict.get('RPA', np.nan)))
if np.isnan(mri_landmarks).any():
    raise RuntimeError('Could not parse T1w sidecar file: "{}"\n\n'
                       'The sidecar file MUST contain a key '
                       '"AnatomicalLandmarkCoordinates" pointing to a '
                       'dict with keys "LPA", "NAS", "RPA". '
                       'Yet, the following structure was found:\n\n"{}"'
                       .format(t1w_json_path, t1w_json))

# The MRI landmarks are in "voxels". We need to convert the to the
# neuromag RAS coordinate system in order to compare the with MEG landmarks
# see also: `mne_bids.write.write_anat`
t1w_path = t1w_json_path.replace('.json', '.nii')
if not op.exists(t1w_path):
    t1w_path += '.gz'  # perhaps it is .nii.gz? ... else raise an error
if not op.exists(t1w_path):
    raise RuntimeError('Could not find the T1 weighted MRI associated '
                       'with "{}". Tried: "{}" but it does not exist.'
                       .format(t1w_json_path, t1w_path))
t1_nifti = nib.load(t1w_path)
# Convert to MGH format to access vox2ras method
t1_mgh = nib.MGHImage(t1_nifti.dataobj, t1_nifti.affine)

# now extract transformation matrix and put back to RAS coordinates of MRI
vox2ras_tkr = t1_mgh.header.get_vox2ras_tkr()
mri_landmarks = apply_trans(vox2ras_tkr, mri_landmarks)
mri_landmarks = mri_landmarks * 1e-3

# Get MEG landmarks from the raw file
# _, ext = _parse_ext(bids_fname)
# if extra_params is None:
#     extra_params = dict()
#     if ext == '.fif':
#         extra_params = dict(allow_maxshield=True)

raw =  mne.io.read_raw_fif(meg_path)
meg_coords_dict = _extract_landmarks(raw.info['dig'])
meg_landmarks = np.asarray((meg_coords_dict['LPA'],
                            meg_coords_dict['NAS'],
                            meg_coords_dict['RPA']))

# Given the two sets of points, fit the transform
trans_fitted = fit_matched_points(src_pts=meg_landmarks,
                                  tgt_pts=mri_landmarks)
trans = mne.transforms.Transform(fro='head', to='mri', trans=trans_fitted)

#Apply necessary coordinate swaps 
#FIX 
import copy
r0 = copy.deepcopy(trans['trans'][0,:])
r1 = copy.deepcopy(trans['trans'][1,:])
r2 = copy.deepcopy(trans['trans'][2,:])

trans['trans'][0,:]=r1
trans['trans'][1,:]=r0
trans['trans'][2,:]*=-1

trans.save(os.path.splitext(t1w_path)[0]+'-trans.fif')

