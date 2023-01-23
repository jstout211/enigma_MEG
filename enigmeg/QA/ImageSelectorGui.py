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
#%%


import PySimpleGUI as sg
import os, os.path as op
from mne_bids import BIDSPath
from mne.viz import Brain
import glob

sg.set_options(font='Courier 18')

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

import pandas as pd

class QA_layout():
    def __init__(self, image_list=None, qa_type=None, grid_size=GRID_SIZE):
        self.idx=0
        self.QA_type = qa_type
        self.grid_size=grid_size
        self.resize_xy = (600,600)
        self.image_list=image_list
        self.layout = self.create_layout()
        self.update_ignore=(len(self.layout),)
        # self.update_layout()
        self.ratings=pd.DataFrame(image_list, columns=['fname'])
        self.ratings['rating']=None

    def encode_image(self,fname):
        with open(fname, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string 
    
    def create_layout(self):
        layout = [  [sg.Text(f'QA: {self.QA_type}')]]
        self.idx=0
        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                # image_ = self.encode_image(self.image_list[self.idx])
                image_ = resize_image(self.image_list[self.idx], resize=self.resize_xy) #image_)
                button_name = self.image_list[self.idx]
                row.append(sg.Button(image_data=image_, border_width=5, key=f'-{str(self.idx)}-', 
                                     image_size=self.resize_xy, expand_x=True, expand_y=True))
                self.idx+=1
            layout.append(row)
        layout.append([sg.Button('PREV'), sg.Button('NEXT'), sg.Button('EXIT')])
        return layout
        
    def update_layout(self): #layout=None, imagelist=None, idx=None):
        '''After prev/next button press - update with new images'''
        for row_idx, row in enumerate(self.layout):
            # Bottom buttons should be ignored
            if row_idx in self.update_ignore:
                continue
            for button in row:
                if self.idx<len(self.image_list):
                    tmp_ = resize_image(self.image_list[self.idx], resize=self.resize_xy)
                    button.update(image_data=tmp_)
                else:
                    continue
                    # button.update(image_data=None)
                          
    

image_list = glob.glob('/home/jstout/Pictures/*.png')


QA_type = 'Freesurfer Recon'
# layout=create_layout(QA_type, grid_size=GRID_SIZE)
QA = QA_layout(image_list, qa_type=QA_type, grid_size=(2,2))


#%%
window = sg.Window(QA_type, QA.layout, resizable=True, auto_size_buttons=True,
                   scaling=True) 
    

try:
    idx=QA.idx
    while True:             # Event Loop
        event, values = window.read()
        print(event, values)
        if event in (sg.WIN_CLOSED, 'EXIT'):
            break
        if event=='NEXT':
            if idx+(GRID_SIZE[0]*GRID_SIZE[1]) < len(image_list):
                idx+=(GRID_SIZE[0]*GRID_SIZE[1])
                # window[f'-{idx}-'].Update(image_data=image_list[idx])
            window['-2-'].Update(image_data=image_list[4])
                # QA.idx = idx
                # QA.update_layout()
            
        if event=='PREV':
            if idx-(GRID_SIZE[0]*GRID_SIZE[1]) < 0:
                idx=0
            else:
                idx-(GRID_SIZE[0]*GRID_SIZE[1])
        # update_layout(layout)
    window.close()
except:
    print('Errored - Closing')
    window.close()
#%%

import PIL.Image
import io
import base64

def resize_image(image_path, resize=(200,200)): 
    if isinstance(image_path, str):
        img = PIL.Image.open(image_path)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(image_path)))
        except Exception as e:
            data_bytes_io = io.BytesIO(image_path)
            img = PIL.Image.open(data_bytes_io)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)))#, PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    # del img
    # buffered=io.BytesIO()
    return base64.b64encode(bio.getvalue()) #bio.getvalue()

