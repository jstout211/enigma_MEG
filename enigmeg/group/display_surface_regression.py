#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:39:32 2020

@author: stoutjd

Modified example from the pysurfer website


"""




import os
import numpy as np
import nibabel as nib
from mne.viz import Brain

import pandas as pd

print(__doc__)


# dframe=pd.read_csv('/home/stoutjd/data/ENIGMA/Prelim_coeffs_Age2.csv')
dframe = pd.read_csv('/home/stoutjd/CAMCAN_ageValpha/Alpha_vs_age.csv')

os.environ['SUBJECTS_DIR'] = '/home/stoutjd/data/BIDS_nih/'
subject_id = "fsaverage"
hemi = "lh"
surf = "inflated"

"""
Bring up the visualization.
"""
brain = Brain(subject_id, hemi, surf, background="white")

"""
Read in the automatic parcellation of sulci and gyri.
"""
aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                          subject_id, "label",
                          hemi + ".aparc.annot") #a2009s.annot")
labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

"""
Make a random vector of scalar data corresponding to a value for each region in
the parcellation.

"""
rs = np.random.RandomState(4)
roi_data = rs.uniform(.5, .8, size=len(names))

names2=[i.decode() for i in names]
names2.remove('corpuscallosum')
names2.remove('unknown')


roi_data = np.zeros(76)


for idx,name in enumerate(names2):
	#idx+=-1 #Accomodate for -1 being unknown
	dframe.loc[idx, 'name']=name
	roi_data[idx]=dframe.loc[idx, 'coeff']
	
	

	 


"""
Make a vector containing the data point at each vertex.
"""
vtx_data = roi_data[labels]

"""
Handle vertices that are not defined in the annotation.
"""
vtx_data[labels == -1] = 0

thresh=.001
vtx_data[np.abs(vtx_data)<thresh]=0

"""
Display these values on the brain. Use a sequential colormap (assuming
these data move from low to high values), and add an alpha channel so the
underlying anatomy is visible.
"""
#brain.add_data(vtx_data, -.001, 0.001, colormap="coolwarm", alpha=1)
brain.add_data(vtx_data, fmin=-0.01, fmax=0.01, colormap="coolwarm", alpha=1) 
