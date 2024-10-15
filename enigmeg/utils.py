#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:17:15 2024

@author: nugenta
"""


import mne
import mne_bids
import json
from pathlib import Path
import numpy as np

def get_head_mri_trans_bti(bids_path, t1_bids_path=None, fs_subject=None, kind=None, 
                           fs_subjects_dir=None):
      
    nib = mne_bids.utils._import_nibabel("get a head to MRI transform")

    # check root available
    meg_bids_path = bids_path.copy()
    del bids_path

    # if the bids_path is underspecified, only get info for MEG data
    if meg_bids_path.datatype is None:
        meg_bids_path.datatype = "meg"
    if meg_bids_path.suffix is None:
        meg_bids_path.suffix = "meg"
 
    t1w_bids_path = (
        (meg_bids_path if t1_bids_path is None else t1_bids_path)
        .copy()
        .update(datatype="anat", suffix="T1w", task=None)
    )
    t1w_json_path = mne_bids.path._find_matching_sidecar(
        bids_path=t1w_bids_path, extension=".json", on_error="ignore"
        )
    del t1_bids_path

    if t1w_json_path is not None:
        t1w_json_path = Path(t1w_json_path)

    if t1w_json_path is None or not t1w_json_path.exists():
        raise FileNotFoundError(
            f"Did not find T1w JSON sidecar file, tried location: " f"{t1w_json_path}"
        )
    
    for extension in (".nii", ".nii.gz"):
        t1w_path_candidate = t1w_json_path.with_suffix(extension)
        if t1w_path_candidate.exists():
            t1w_bids_path = mne_bids.path.get_bids_path_from_fname(fname=t1w_path_candidate)
            break

    # Get MRI landmarks from the JSON sidecar
    t1w_json = json.loads(t1w_json_path.read_text(encoding="utf-8"))
    mri_coords_dict = t1w_json.get("AnatomicalLandmarkCoordinates", dict())

    # landmarks array: rows: [LPA, NAS, RPA]; columns: [x, y, z]
    suffix = f"_{kind}" if kind is not None else ""
    mri_landmarks = np.full((3, 3), np.nan)
    for landmark_name, coords in mri_coords_dict.items():
        if landmark_name.upper() == ("LPA" + suffix).upper():
            mri_landmarks[0, :] = coords
        elif landmark_name.upper() == ("RPA" + suffix).upper():
            mri_landmarks[2, :] = coords
        elif (
            landmark_name.upper() == ("NAS" + suffix).upper()
            or landmark_name.lower() == ("nasion" + suffix).lower()
        ):
            mri_landmarks[1, :] = coords
        else:
            continue
        
    fs_t1_path = Path(fs_subjects_dir) / fs_subject / "mri" / "T1.mgz"
    if not fs_t1_path.exists():
        raise ValueError(
            f"Could not find {fs_t1_path}. Consider running FreeSurfer's "
            f"'recon-all` for subject {fs_subject}."
        )
    fs_t1_mgh = nib.load(str(fs_t1_path))
    t1_nifti = nib.load(str(t1w_bids_path.fpath))

    # Convert to MGH format to access vox2ras method
    t1_mgh = nib.MGHImage(t1_nifti.dataobj, t1_nifti.affine)

    # convert to scanner RAS
    mri_landmarks = mne.transforms.apply_trans(t1_mgh.header.get_vox2ras(), mri_landmarks)

    # convert to FreeSurfer T1 voxels (same scanner RAS as T1)
    mri_landmarks = mne.transforms.apply_trans(fs_t1_mgh.header.get_ras2vox(), mri_landmarks)

    # now extract transformation matrix and put back to RAS coordinates of MRI
    vox2ras_tkr = fs_t1_mgh.header.get_vox2ras_tkr()
    mri_landmarks = mne.transforms.apply_trans(vox2ras_tkr, mri_landmarks)
    mri_landmarks = mri_landmarks * 1e-3

    bti_pdf_patterns = ["0", "c,rf*", "hc,rf*", "e,rf*"]
    pdf_list = []
    bids_raw_folder = meg_bids_path.directory / f"{meg_bids_path.basename}"
    for pattern in bti_pdf_patterns:
         pdf_list += sorted(bids_raw_folder.glob(pattern))         
    raw_path = pdf_list[0]
    config_path = bids_raw_folder / "config"
    
    raw = mne.io.read_raw_bti(raw_path)

    pos = raw.get_montage().get_positions()
    meg_landmarks = np.asarray((pos["lpa"], pos["nasion"], pos["rpa"]))

    # Given the two sets of points, fit the transform
    trans_fitted = mne.coreg.fit_matched_points(src_pts=meg_landmarks, tgt_pts=mri_landmarks)
    trans = mne.transforms.Transform(fro="head", to="mri", trans=trans_fitted)
    return trans


