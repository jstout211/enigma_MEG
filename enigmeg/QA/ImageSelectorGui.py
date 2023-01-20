#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:13:00 2023

@author: jstout
"""

import PySimpleGUI as sg

sg.theme('DarkAmber')   # Add a touch of color

# All the stuff inside your window.
layout = [  [sg.Text('Enter the MEG BIDs directory')],
            [sg.Text('BIDS DIR:'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')] ]

def QA_images():
# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    print('You entered ', values[0])

window.close()

# =============================================================================
# Image as button
# =============================================================================

import PySimpleGUI as sg
import os, os.path as op
from mne_bids import BIDSPath
from mne.viz import Brain
import glob

import base64
image = '/home/jstout/Desktop/Plot1.png'
subject = 'DEC105'
session = '1'
run = '01'

bids_root = '/fast/oberman_test/BIDS'
deriv_root = op.join(bids_root, 'derivatives', 'ENIGMA_MEG')
subjects_dir = op.join(bids_root, 'derivatives','freesurfer', 'subjects')
deriv_path = BIDSPath(root=deriv_root, 
                      check=False,
                      subject=subject,
                      session=session,
                      run=run,
                      datatype='meg'
                      )
QA_path = deriv_path.directory / 'ENIGMA_QA'


def save_brain_images(deriv_path, hemi=None, qa_subdir='ENIGMA_QA'):
    # out_path = deriv_path.directory / qa_subdir
    out_fname = deriv_path.copy().update(description=f'QAfsrecon',
                                         suffix=hemi,
                                         extension='.png').fpath
    if not op.exists(out_fname):
        brain = Brain(subject='sub-'+deriv_path.subject,
             subjects_dir=subjects_dir,
             hemi=hemi,
             title=deriv_path.subject
             )
        brain.save_image(filename=out_fname)
        brain.close()
    return out_fname

def merge_and_convert():
    ...

lh_brain_fname = save_brain_images(deriv_path, hemi='lh')
rh_brain_fname = save_brain_images(deriv_path, hemi='rh')



## Find all completed datasets 
# If completed load filename info
def compile_completed(deriv_path):
    '''Identify all subject QA files'''
    tmp_ = op.join(deriv_path.root,
                   'sub-*','ses-*','meg', '*QAfsrecon*lh.png'  #FIX - Only lh hemi !!
                   )
    return glob.glob(tmp_)


grid_images = compile_completed(deriv_path)



GRID_SIZE = (6,4)
## Define Grid
# Make a def to loop over images and 


## Create Left/Right Arrows to loop through all subjects
# Loop numbers are total / (GRID_SIZE[0]* GRID_SIZE[1])

## If clicked - write class identifier as bad
# If clicked a second time - undo

# At the end of loop save all bads into QA folders





def encode_image(fname):
    with open(fname, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string 

def create_layout(QA_type, grid_size=(6,4)):
    layout = [  [sg.Text(f'QA: {QA_type}')]]
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            row.append(sg.Button(''))
        layout.append(row)
    layout.append([sg.Button('PREV'), sg.Button('NEXT'), sg.Button('EXIT')])
    return layout
    
def update_layout(layout):
    


QA_type = 'Freesurfer Recon'
layout=create_layout(QA_type, grid_size=GRID_SIZE)
window = sg.Window(QA_type, layout, no_titlebar=True)
    

while True:             # Event Loop
    event, values = window.read()
    print(event, values)
    if event in (sg.WIN_CLOSED, 'EXIT'):
        break
    if event=='NEXT':
        if idx+(GRID_SIZE[0]*GRID_SIZE[1]) < len(image_set):
            idx+=(GRID_SIZE[0]*GRID_SIZE[1])
    if event=='PREV':
        if idx-(GRID_SIZE[0]*GRID_SIZE[1]) < 0:
            idx=0
        else:
            idx-(GRID_SIZE[0]*GRID_SIZE[1])
    update_layout(layout)
window.close()

