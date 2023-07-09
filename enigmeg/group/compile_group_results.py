#!/usr/bin/env python

import glob
import os, os.path as op
import shutil
import sys
topdir=os.getcwd() 
print(topdir)
if not op.exists('./ENIGMA_MEG'):
   print('You must be in the derivatives folder with ENIGMA_MEG present')
   raise
os.chdir('ENIGMA_MEG')

label_spec_list = glob.glob('**/label_spectra.csv', recursive=True)
band_rel_list = glob.glob('**/Band_rel_power.csv', recursive=True)

group_dir = op.join(os.getcwd(),'GROUP')
os.makedirs(group_dir, exist_ok=True)

#os.chdir(topdir)
print('Copying label spectra to GROUP')
for fname in label_spec_list:
    subjid = fname.split('/')[0]
    outfname = op.join(group_dir, f'{subjid}_label_spectra.csv')
    shutil.copy(fname, outfname)

print('Copying Band_rel_power to GROUP')
for fname in band_rel_list:
    subjid = fname.split('/')[0]
    outfname = op.join(group_dir, f'{subjid}_band_rel_power.csv')
    shutil.copy(fname, outfname)

print(f'There are {len(label_spec_list)} csv files copied')
