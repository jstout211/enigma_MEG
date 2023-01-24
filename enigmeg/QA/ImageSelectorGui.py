#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:13:00 2023

@author: jstout
"""

import PySimpleGUI as sg
import os, os.path as op
from mne_bids import BIDSPath
from mne.viz import Brain
import glob
import PIL.Image
import io
import base64
import copy
import enigmeg
import logging

NULL_IMAGE = op.join(enigmeg.__path__[0], 'QA', 'Null.png')
sg.set_options(font='Courier 18')
QA_type='FSrecon'


image = '/home/jstout/Desktop/Plot1.png'
subject = 'DEC105'
session = '1'
run = '01'
GRID_SIZE=(2,2)

bids_root = '/fast/oberman_test/BIDS'
deriv_root = op.join(bids_root, 'derivatives', 'ENIGMA_MEG')
subjects_dir = op.join(bids_root, 'derivatives','freesurfer', 'subjects')

# Set up logging
log = op.join(deriv_root, 'enigma_QA_logfile.txt')
logging.basicConfig(filename=log, encoding='utf-8', level=logging.DEBUG, 
                    format='%(levelname)s:%(message)s')

format='%(levelname)s:%(message)s'

# deriv_path = BIDSPath(root=deriv_root, 
#                       check=False,
#                       subject=subject,
#                       session=session,
#                       run=run,
#                       datatype='meg'
#                       )
# QA_path = deriv_path.directory / 'ENIGMA_QA'


# def save_brain_images(deriv_path, hemi=None, qa_subdir='ENIGMA_QA'):
#     # out_path = deriv_path.directory / qa_subdir
#     out_fname = deriv_path.copy().update(description=f'QAfsrecon',
#                                          suffix=hemi,
#                                          extension='.png').fpath
#     if not op.exists(out_fname):
#         brain = Brain(subject='sub-'+deriv_path.subject,
#              subjects_dir=subjects_dir,
#              hemi=hemi,
#              title=deriv_path.subject
#              )
#         brain.save_image(filename=out_fname)
#         brain.close()
#     return out_fname

# def merge_and_convert():
#     ...

# lh_brain_fname = save_brain_images(deriv_path, hemi='lh')
# rh_brain_fname = save_brain_images(deriv_path, hemi='rh')



# ## Find all completed datasets 
# # If completed load filename info
# def compile_completed(deriv_path):
#     '''Identify all subject QA files'''
#     tmp_ = op.join(deriv_path.root,
#                    'sub-*','ses-*','meg', '*QAfsrecon*lh.png'  #FIX - Only lh hemi !!
#                    )
#     return glob.glob(tmp_)


# grid_images = compile_completed(deriv_path)



def resize_image(image_path, resize=(200,200)): 
    '''
    Configure images to be used as PySimpleGUI buttons

    Parameters
    ----------
    image_path : path string
        Path to input image
    resize : Tuple, optional
        Pixel X Pixel Dimension. The default is (200,200).

    Returns
    -------
    Base64 encoded string
        Base64 string for use as button image.

    '''
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
        img = img.resize((int(cur_width*scale), int(cur_height*scale)))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue())

# Create a list of objects unique to the subjects
class sub_qa_info():
    '''Store info on the status of subject QA results'''
    def __init__(self, idx=None, fname=None, qa_type='FSrecon', log=None):
        self.idx=idx
        self.fname = fname
        self.qa_type=qa_type
        self.status = self.check_status()  
        self.subject = self.get_subjid()
    
    ############## VERIFY #####################
    def set_status(self):
        '''Set up toggle for GOOD/BAD'''
        if self.status=='Unchecked':
            self.status = 'BAD'
        if self.status=='GOOD':
            self.status = 'BAD'
        if self.status=='BAD':
            self.status = 'GOOD'
    
    def log_status(self):
        '''Check if this has been previously set in the log'''
        return 
        
    #!!! FIX Need to make a QA list that is queried to determine good/bad/unchecked
    def check_status(self):
        if not hasattr(self, 'status'):
            return 'Unchecked'
        else:
            return self.status
    ###########################################
    
    def get_subjid(self):
        base = op.basename(self.fname)
        try:
            return base.split('_')[0][4:]
        except:
            return None
    


def create_window_layout(image_list=None, sub_obj_list=None, qa_type=None, 
                         grid_size=GRID_SIZE, frame_start_idx=0, 
                         resize_xy=(600,600)):
    layout = [[sg.Text(f'QA: {qa_type}')]]
    frame_end_idx=frame_start_idx+grid_size[0]*grid_size[1]
    current_idx = copy.deepcopy(frame_start_idx)
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            if current_idx >= len(image_list):
                image_ = resize_image(NULL_IMAGE, resize=resize_xy)
                row.append(sg.Button(image_data=image_, border_width=5, key=None, 
                                 image_size=resize_xy, expand_x=True, expand_y=True))
            else:
                image_ = resize_image(image_list[current_idx], resize=resize_xy)
                button_name = sub_obj_list[current_idx].subject
                row.append(sg.Button(image_data=image_, border_width=5, key=sub_obj_list[current_idx], 
                                 image_size=resize_xy, expand_x=True, expand_y=True))
            current_idx +=1
        layout.append(row)
    layout.append([sg.Button('PREV'), sg.Button('NEXT'), sg.Button('EXIT')])
    window = sg.Window(QA_type, layout, resizable=True, auto_size_buttons=True,
                   scaling=True)
    return window

# =============================================================================
# GUI component
# =============================================================================
image_list = glob.glob('/home/jstout/Pictures/*.png')    
sub_obj_list = [sub_qa_info(i, fname) for i,fname in enumerate(image_list)]

GRID_SIZE=(2,2)
idx=0
window = create_window_layout(image_list, sub_obj_list, qa_type=QA_type, 
                              grid_size=GRID_SIZE,
                              frame_start_idx=idx)

while True:             # Event Loop
    # print(idx)
    event, values = window.read()
    # print(event, values)
    if event in (sg.WIN_CLOSED, 'EXIT'):
        break
    if event=='NEXT':
        if idx+GRID_SIZE[0]*GRID_SIZE[1] < len(image_list):
            idx+= GRID_SIZE[0]*GRID_SIZE[1]
            modify_frame = True
        else:
            print('End of Frame')
    if event=='PREV':
        if idx-(GRID_SIZE[0]*GRID_SIZE[1]) < 0:
            idx=0
        else:
            idx-=(GRID_SIZE[0]*GRID_SIZE[1])
            modify_frame = True
    if type(event) is sub_qa_info:
        event.set_status()
    if modify_frame == True:
        window.close()
        window=create_window_layout(image_list, sub_obj_list,
                                        qa_type=QA_type, grid_size=GRID_SIZE,
                                        frame_start_idx=idx)
        modify_frame = False
for subj_qa in sub_obj_list:
    print(f'{subj_qa.subject}:{subj_qa.fname}:{subj_qa.status}')
window.close()


#%%
# =============================================================================
# TESTs currently (need to move to subdirectory)
# =============================================================================
def test_sub_qa_info():
    qai = sub_qa_info(idx=10, fname='/home/jstout/sub-ON10001_task-yadayada_session-1_meg.ds')
    assert qai.subject == 'ON10001'
    assert qai.status == 'Unchecked'

tmp = test_sub_qa_info()


#%%
#import pandas as pd

# class QA_layout():
#     def __init__(self, image_list=None, qa_type=None, grid_size=GRID_SIZE):
#         self.idx=0
#         self.QA_type = qa_type
#         self.grid_size=grid_size
#         self.resize_xy = (600,600)
#         self.image_list=image_list
#         self.layout = self.create_layout()
#         self.update_ignore=(len(self.layout),)
#         # self.update_layout()
#         self.ratings=pd.DataFrame(image_list, columns=['fname'])
#         self.ratings['rating']=None

#     def encode_image(self,fname):
#         with open(fname, "rb") as image_file:
#             encoded_string = base64.b64encode(image_file.read())
#         return encoded_string 
    
#     def create_layout(self):
#         layout = [  [sg.Text(f'QA: {self.QA_type}')]]
#         self.idx=0
#         for i in range(self.grid_size[0]):
#             row = []
#             for j in range(self.grid_size[1]):
#                 # image_ = self.encode_image(self.image_list[self.idx])
#                 image_ = resize_image(self.image_list[self.idx], resize=self.resize_xy) #image_)
#                 button_name = self.image_list[self.idx]
#                 row.append(sg.Button(image_data=image_, border_width=5, key=f'-{str(self.idx)}-', 
#                                      image_size=self.resize_xy, expand_x=True, expand_y=True))
#                 self.idx+=1
#             layout.append(row)
#         layout.append([sg.Button('PREV'), sg.Button('NEXT'), sg.Button('EXIT')])
#         return layout
        
#     def update_layout(self): #layout=None, imagelist=None, idx=None):
#         '''After prev/next button press - update with new images'''
#         for row_idx, row in enumerate(self.layout):
#             # Bottom buttons should be ignored
#             if row_idx in self.update_ignore:
#                 continue
#             for button in row:
#                 if self.idx<len(self.image_list):
#                     tmp_ = resize_image(self.image_list[self.idx], resize=self.resize_xy)
#                     button.update(image_data=tmp_)
#                 else:
#                     continue
#                     # button.update(image_data=None)
                          
    

# image_list = glob.glob('/home/jstout/Pictures/*.png')


# QA_type = 'Freesurfer Recon'
# # layout=create_layout(QA_type, grid_size=GRID_SIZE)
# QA = QA_layout(image_list, qa_type=QA_type, grid_size=(2,2))

