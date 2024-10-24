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
sfreq = raw.info['sfreq']
evts, evts_id = mne.events_from_annotations(raw)

assert evts[0,-1]==2  #Starts and ends with eyes open
assert evts[-1,-1]==2


for row_idx in range(evts.shape[0]):
    if row_idx % 2 != 0: #force evens
        continue
    if row_idx==len(anot)-1:  #If last row, duration finishes at term of file
        dur = raw.times[-1] - anot[row_idx]['onset']
    else:
        dur = anot[row_idx+1]['onset'] - anot[row_idx]['onset']
    evts[row_idx,1] = dur *sfreq

# Build out the data matrix while subtracting the offset
raw_stack = [raw._data[:, evts[0,0]:(evts[0,0]+evts[0,1])]]
stack_idx = 0
for row_idx in range(evts.shape[0]):
    if row_idx % 2 != 0: #force evens
        continue
    if row_idx == 0: #force start at 2
        continue
    print(row_idx)
    tmp = raw._data[:, evts[row_idx,0]:(evts[row_idx,0]+evts[row_idx,1])]
    raw_stack.append(tmp - tmp[:,0][:,np.newaxis]+raw_stack[-1][:,-1][:,np.newaxis])
    
   
raw_array = np.concatenate(raw_stack, axis=-1)    
    
#Make mne object from concatenated array
raw_hack = mne.io.RawArray(raw_array, info=raw.info)


# Add BAD designation to all breaks in data
new_evts = evts[::2,:]
new_evts[0,0]=0
current_start=0
for row_idx in range(new_evts.shape[0]-1):
    print(row_idx)
    current_start+=(new_evts[row_idx, 1] / sfreq)
    new_evts[row_idx, 0]=current_start -1 


annots = mne.Annotations(new_evts[:,0], 2, description='BAD_seg')
raw_hack.set_annotations(annots)

raw_hack.save(fname.replace('.ds','.fif'))
