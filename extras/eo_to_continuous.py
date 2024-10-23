#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:12:27 2024

@author: jstout
"""

import os, os.path as op
import mne 
import glob
import numpy as np


import sys
fname = sys.argv[1]


raw = mne.io.read_raw_ctf(fname, preload=True, system_clock='ignore')

# Below is some semi-crappy code - seems to work
anot = raw.annotations
dur1 = anot[1]['onset'] - anot[0]['onset']
dur2 = anot[3]['onset'] - anot[2]['onset']
dur3 = raw.times[-1] - anot[4]['onset']

sfreq = raw.info['sfreq']
evts, evts_id = mne.events_from_annotations(raw)
evts[0,1] = dur1 * sfreq
evts[2,1] = dur2 * sfreq
evts[4,1] = dur3 * sfreq

raw_stack = [raw._data[:, evts[0,0]:(evts[0,0]+evts[0,1])]]
tmp = raw._data[:, evts[2,0]:(evts[2,0]+evts[2,1])]
raw_stack.append(tmp - tmp[:,0][:,np.newaxis]+raw_stack[0][:,-1][:,np.newaxis])
tmp2 = raw._data[:, evts[4,0]:(evts[4,0]+evts[4,1])]
raw_stack.append(tmp2-tmp2[:,0][:,np.newaxis]+ raw_stack[1][:,-1][:,np.newaxis]  )    
raw_array = np.concatenate(raw_stack, axis=-1)    
    
#Make mne object from concatenated array
raw_hack = mne.io.RawArray(raw_array, info=raw.info)

new_evts = np.zeros([2,3])
new_evts[0,0] = dur1 - 1  #Start a second before
new_evts[1,0] = dur1+dur2 - 1

annots = mne.Annotations(new_evts[:,0], 2, description='BAD_seg')
raw_hack.set_annotations(annots)

raw_hack.save(fname.replace('.ds','.fif'))
