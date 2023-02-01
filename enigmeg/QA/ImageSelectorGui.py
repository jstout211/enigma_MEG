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
from PIL import Image, ImageDraw

# =============================================================================
# Defaults
# =============================================================================
NULL_IMAGE = op.join(enigmeg.__path__[0], 'QA', 'Null.png')
sg.set_options(font='Courier 18')
status_color_dict = {'Unchecked':'grey',
                   'GOOD':'green',
                   'BAD':'red'
                   }
GRID_SIZE=(3,6)
PROJECT = 'ENIGMA_MEG_QA'
SEARCH_DICT = {'FSrecon': f'{PROJECT}/sub-*/meg/*QAfsrecon*.png'}
qa_types = SEARCH_DICT.keys()
# =============================================================================
# Commandline Interface
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root',
                        help='''Location of bids directory used for enigma
                        processing''')
    parser.add_argument('-qa_type',
                        help='''Type of image to lookup''',
                        default='FSrecon')
    args = parser.parse_args()
    bids_root=args.bids_root
    QA_type=args.qa_type
    
deriv_root = op.join(bids_root, 'derivatives')                       
                        

# Set up logging
logfile = op.join(deriv_root, PROJECT, 'enigma_QA_logfile.txt')
if op.exists(logfile):
    with open(logfile) as f:
        history_log = f.readlines()
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.info("REVIEW_START")

# =============================================================================
# Generate QA images 
# =============================================================================
def generate_QA_images(bids_root, subject=None, session=None, 
                       run='1'):
    deriv_root = op.join(bids_root, 'derivatives', 'ENIGMA_MEG')
    deriv_root_qa = op.join(bids_root, 'derivatives', 'ENIGMA_MEG_QA')
    subjects_dir = op.join(bids_root, 'derivatives','freesurfer', 'subjects')
    deriv_path = BIDSPath(root=deriv_root, 
                          check=False,
                          subject=subject,
                          session=session,
                          run=run,
                          datatype='meg'
                          )
    qa_path = deriv_path.copy().update(root=deriv_root_qa, description='QAfsrecon',
                                       extension='.png',  suffix='lh').fpath   #!!! FIX Hemi needs to be a declared var
    lh_brain_fname = save_brain_images(deriv_path, hemi='lh', out_fname=qa_path)

def save_brain_images(deriv_path, hemi=None, out_fname=None):
    # out_path = deriv_path.directory / qa_subdir
    # out_fname = deriv_path.copy().update(description=f'QAfsrecon',
    #                                       suffix=hemi,
    #                                       extension='.png').fpath
    os.makedirs(op.dirname(out_fname), exist_ok=True)
    if not op.exists(out_fname):
        brain = Brain(subject='sub-'+deriv_path.subject,
              subjects_dir=subjects_dir,
              hemi=hemi,
              title=deriv_path.subject
              )
        brain.save_image(filename=out_fname)
        brain.close()
    return out_fname

for subj in glob.glob(op.join(bids_root, 'sub-*')):
    subj = op.basename(subj)[4:]
    generate_QA_images(bids_root, subject=subj, session=None, 
                       run='1')
    
# =============================================================================
# Functions to Process images and set current status
# =============================================================================
def resize_image(image_path, resize=(200,200), status=None, 
                 text_val=None): 
    '''
    Configure images to be used as PySimpleGUI buttons
    Add a color bar at the bottom of the image designating the status
    Write the Subject ID into colorbar

    Parameters
    ----------
    image_path : path string
        Path to input image
    resize : Tuple, optional
        Pixel X Pixel Dimension. The default is (200,200).
    status : str
        Used in the STATUS_COLOR_DICT to extract a background color
        Must be one of the following Unchecked / GOOD / BAD 
    text_val : str
        Written in colorbar - typically the subject ID

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
    status_color = status_color_dict[status]
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)))
        img_draw = ImageDraw.Draw(img)
        #Add text box
        img_draw.rectangle((0, 580, 600, 600), outline=None, fill=status_color)
        img_draw.text((20, 590), text_val, align='center', fill='black')
        
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue())

class sub_qa_info():
    '''Store info on the status of subject QA results'''
    def __init__(self, idx=None, fname=None, qa_type='FSrecon', log=None, 
                 resize_xy=(600,600)):
        self.idx=idx
        self.fname = fname
        self.qa_type=qa_type
        self.status = self.check_status()  
        self.subject = self.get_subjid()
        self.image_r = resize_image(self.fname, resize=resize_xy, 
                                    status=self.status, text_val=self.subject)
    
    ############## VERIFY #####################
    def set_status(self):
        '''Set up toggle for Unchecked/GOOD/BAD'''
        if self.status=='Unchecked':
            self.status = 'GOOD'
        elif self.status=='GOOD':
            self.status = 'BAD'
        elif self.status=='BAD':
            self.status = 'Unchecked'
    
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
    
def load_logfile(logfile):
    '''
    Load the file and check for subject entries.  Make a dictionary that can 
    be queried during qa_info creation to load the status

    Returns
    -------
    None.

    '''
    with open(logfile) as f:
        history = f.readlines()
    return history

def parse_history(history):  #!!!FIX Empty function
    '''
    Identify the last time the log was initiated
    Generate a dictionary with subject ids to readout the status
    

    Parameters
    ----------
    history : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    return 


def write_logfile(obj_list):
    '''
    Loop over all subjects in list and write the status to the logfile

    Parameters
    ----------
    obj_list : [sub_qa_info]
        List of subject info objects.
        Each entry has a status tag that will be written to the logfile

    Returns
    -------
    None.

    '''
    for obj in obj_list:
        logging.info(f"SUBJECT:{obj.subject}:STATUS:{obj.status}")

def create_window_layout(sub_obj_list=None, qa_type=None, 
                         grid_size=GRID_SIZE, frame_start_idx=0, 
                         resize_xy=(600,600)):
    layout = [[sg.Text(f'QA: {qa_type}')]]
    frame_end_idx=frame_start_idx+grid_size[0]*grid_size[1]
    current_idx = copy.deepcopy(frame_start_idx)
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            if current_idx >= len(sub_obj_list):
                image_ = resize_image(NULL_IMAGE, resize=resize_xy, 
                                      status='Unchecked', text_val='')
                row.append(sg.Button(image_data=image_, border_width=5, key=None, 
                                 image_size=resize_xy, expand_x=True, expand_y=True))
            else:
                image_ = resize_image(sub_obj_list[current_idx].image_r,
                                      resize=resize_xy, 
                                      status=sub_obj_list[current_idx].status,
                                      text_val=sub_obj_list[current_idx].subject
                                      )
                button_name = sub_obj_list[current_idx].subject
                row.append(sg.Button(image_data=image_, border_width=5, 
                                     key=sub_obj_list[current_idx],
                                     image_size=resize_xy, expand_x=True, 
                                     expand_y=True))
            current_idx +=1
        layout.append(row)
    layout.append([sg.Button('PREV'), sg.Button('NEXT'), sg.Button('EXIT'), 
                   sg.Button('SAVE')])
    window = sg.Window(QA_type, layout, resizable=True, auto_size_buttons=True,
                   scaling=True)
    return window

# =============================================================================
# GUI component
# =============================================================================
image_list = glob.glob(op.join(deriv_root, SEARCH_DICT['FSrecon']))
sub_obj_list = [sub_qa_info(i, fname) for i,fname in enumerate(image_list)]

idx=0
window = create_window_layout(sub_obj_list, qa_type=QA_type, 
                              grid_size=GRID_SIZE,
                              frame_start_idx=idx)
modify_frame=False
while True:             # Event Loop
    # print(idx)
    event, values = window.read()
    # print(event, values)
    if event in (sg.WIN_CLOSED, 'EXIT'):
        break
    if event=='NEXT':
        if idx+GRID_SIZE[0]*GRID_SIZE[1] < len(sub_obj_list):
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
    if event=='SAVE':
        write_logfile(sub_obj_list)
    if type(event) is sub_qa_info:
        event.set_status()
        image_ = resize_image(event.image_r,
                      resize=(600,600) ,  #!!!FIX - should be a variable
                      status=event.status,
                      text_val=event.subject
                      )
        window[event].update(image_data=image_) 
    if modify_frame == True:
        window.close()
        window=create_window_layout(sub_obj_list,
                                        qa_type=QA_type, grid_size=GRID_SIZE,
                                        frame_start_idx=idx)
        modify_frame = False

window.close()
logging.info("REVIEW_FINISH")

#%%
# def test_resize_image():
#     image_path = '/fast/tmp_QA/BIDS_stringaris/derivatives/ENIGMA_MEG_QA/sub-24208/meg/sub-24208_run-1_desc-QAfsrecon_lh.png'
#     resize=(200,200)
#     status=None
#     PIL.ImageDraw.Draw.rectangle()
#     subjid = 'TEST_SUBJ'
#     status = 'red'    
#     # from PIL import Image, ImageDraw

#     canvas = Image.new('RGB', resize, 'white')
#     img_draw = ImageDraw.Draw(canvas)
#     img_draw.rectangle((0, 580, 600, 600), outline=None, fill=status)
#     img_draw.text((20, 590), subjid, align='center', fill='black', spacing=10)
#     canvas.show()
  

# =============================================================================
# TESTs currently (need to move to subdirectory)
# =============================================================================
# def test_sub_qa_info():
#     qai = sub_qa_info(idx=10, fname='/home/jstout/sub-ON10001_task-yadayada_session-1_meg.ds')
#     assert qai.subject == 'ON10001'
#     assert qai.status == 'Unchecked'

# tmp = test_sub_qa_info()

# =============================================================================
# Test extra
# =============================================================================
# image = '/home/jstout/Desktop/Plot1.png'
# subject = '23520'
# session = '1'
# run = '01'
# GRID_SIZE=(2,2)

