#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:56:04 2020

@author: stoutjd
"""

import mne
filename= '/home/stoutjd/hv_proc/MEG/APBWVFAR_airpuff_20200122_05.ds'
raw = mne.io.read_raw_ctf(filename, preload=True)
raw.resample(300)
raw.notch_filter([60,120])

events = mne.events_from_annotations(raw)


stims = events[0][events[0][:,2]==2]

epochs = mne.Epochs(raw, stims, tmin=-0.1, tmax=0.4)
epochs = epochs.apply_baseline()


noise_cov = mne.compute_covariance(epochs, tmin=-0.1, tmax=0)
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=0.3)

bem = mne.read_bem_solution('/home/stoutjd/data/DEMO_ENIGMA/outputs/APBWVFAR_fs_ortho/bem_sol-sol.fif')
src = mne.read_source_spaces('/home/stoutjd/data/DEMO_ENIGMA/outputs/APBWVFAR_fs_ortho/source_space-src.fif')
transfile='/home/stoutjd/data/ENIGMA/transfiles/APBWVFAR-trans.fif'
trans = mne.read_trans(transfile)

forward = mne.make_forward_solution(epochs.info, trans, src, bem)

filters = mne.beamformer.make_lcmv(epochs.info, forward, data_cov=data_cov,
                                   noise_cov=noise_cov)

evoked = epochs.average()

filters = mne.beamformer.make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=None)

stc = mne.beamformer.apply_lcmv(evoked, filters)



###############  Plotting ##################
import copy, numpy as np
stc2 = copy.deepcopy(stc)
stc2._data = np.abs(stc2._data)
brain=stc2.plot(subject='APBWVFAR_fs_ortho', 
          subjects_dir='/home/stoutjd/data/DEMO_ENIGMA/SUBJECTS', 
          initial_time=.05, clim=dict(kind='value', lims=[.25,.35,.6]), 
          surface ='pial', hemi='both')


for i in labels: brain.add_foci(i.center_of_mass(), coords_as_verts=True, 
                                hemi='lh', color='blue', scale_factor=1.0, 
                                alpha=0.5)

