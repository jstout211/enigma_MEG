#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:17:39 2023

@author: Jeff Stout and Allison Nugent and chatGPT
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
import logging
import matplotlib
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import enigmeg

status_color_dict = {'Unchecked':'grey',
                   'GOOD':'green',
                   'BAD':'red'
                   }
NULL_IMAGE = op.join(enigmeg.__path__[0], 'QA', 'Null.png')
PROJECT = 'ENIGMA_MEG_QA'
    
## class for storing info about the subject QA
    
class sub_qa_info():
    '''Store info on the status of subject QA results'''
    def __init__(self, idx=None, fname=None, qa_type=None, log=None, 
                 resize_xy=(600,600), init_status='Unchecked'):
        self.idx=idx
        self.fname = fname
        self.qa_type = qa_type
        self.status = init_status  
        self.subject = self.get_subjid()
        self.image_r = resize_image(self.fname, resize=resize_xy, 
                                    status=self.status, text_val='')
    
    def button_set_status(self):
        '''Set up toggle for Unchecked/GOOD/BAD'''
        if self.status=='Unchecked':
            self.status = 'GOOD'
        elif self.status=='GOOD':
            self.status = 'BAD'
        elif self.status=='BAD':
            self.status = 'Unchecked'
    
    def get_subjid(self):
        base = op.basename(self.fname)
        try:
            #return base.split('/')[-1].split('_')[0]
            return base
        except:
            return None
        
    def set_status(self, status):
        self.status=status
        
    def set_type(self, qa_type):
        self.qa_type = qa_type
        
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
    new_width, new_height = resize
    scale = min(new_height/cur_height, new_width/cur_width)
    img = img.resize((int(cur_width*scale), int(cur_height*scale)))
    img_draw = ImageDraw.Draw(img)
    img_draw.rectangle((0, 0, cur_width*scale, cur_height*scale), width=8, outline=status_color, fill=None)

    # get the matplotlib font 
    default_font_pattern = matplotlib.font_manager.FontProperties().get_fontconfig_pattern()
    fontpath = matplotlib.font_manager.findfont(default_font_pattern)
    font = ImageFont.truetype(fontpath, size=40)
    img_draw.text((10, 10), text_val, align='center', fill='black')
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue())

def create_window_layout(sub_obj_list=None, qa_type=None, 
                             grid_size=[3,6], frame_start_idx=0, 
                             resize_xy=(100,100)):
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
        layout.append([sg.Button('PREV'), sg.Button('NEXT'), sg.Button('SAVE AND EXIT')])
        window = sg.Window(qa_type, layout, resizable=True, auto_size_buttons=True,
                       scaling=True)
        return window
    
# =============================================================================
# Logfile initialization, parsing, and updating
# =============================================================================

## Initialize History Logfile

def initialize(bids_root, QAtype):
    
    deriv_root = op.join(bids_root, 'derivatives')
    # Set up logging
    logfile = op.join(deriv_root, PROJECT, QAtype+'_QA_logfile.txt')
    return_log=False
    if op.exists(logfile):
        with open(logfile) as f:
            history_log = f.readlines()
        #Strip newlines        
        history_log=[i[:-1] for i in history_log if i[-1:]=='\n']
        return_log=True
    logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.INFO, 
                        format='%(asctime)s::%(levelname)s::%(message)s', filemode='w')
    logging.info("REVIEW_START")
    if return_log==True:
        return history_log
    else:
        return None
    
## Get the last review period

def get_last_review(history_log):
    '''Extract the start and stop of the last review
    Return the log lines from the last review'''
    rev_start_idx=0
    rev_end_idx=0
    for idx, line in enumerate(history_log):
        cond=line.split('INFO::')[-1]
        if cond=='REVIEW_START':
            rev_start_idx=idx
        elif cond=='REVIEW_FINISH':
            rev_end_idx=idx
    last_review=history_log[rev_start_idx+1:rev_end_idx]
    return last_review
        
def get_subject_status(logline_input):
    tmp=logline_input.split(':')
    datetimeval=':'.join(tmp[0:3])
    subject, status, qatype = None, None, None #Preinitialize
    if 'SUBJECT' in tmp:
        sub_idx = tmp.index('SUBJECT')
        subject = tmp[sub_idx+1]
    if 'STATUS' in tmp:
        stat_idx = tmp.index('STATUS')
        status = tmp[stat_idx+1]
    if 'TYPE' in tmp:
        type_idx = tmp.index('TYPE')
        qatype = tmp[type_idx+1]
    return subject, status, qatype

def build_status_dict(review_log):
    '''Loop over lines in review log and extract a dictionary of 
    subject status.'''
    subject_status_dict={}
    for line in review_log:
        subject, status, qatype = get_subject_status(line)
        if subject==None:
            continue
        subject_status_dict[subject]=status
    return subject_status_dict

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
        logging.info(f"SUBJECT:{obj.subject}:TYPE:{obj.qa_type}:STATUS:{obj.status}")
        
## Do the part where we actually run the GUI

def run_gui(sub_obj_list,rows=3,columns=2,imgsize=200, QAtype=None):
    
    idx=0
    GRID_SIZE=[int(rows),int(columns)]
    #print(GRID_SIZE)
    window = create_window_layout(sub_obj_list, qa_type=QAtype, 
                                  grid_size=GRID_SIZE,
                                  frame_start_idx=idx,resize_xy=(imgsize,imgsize))
    modify_frame=False
    while True:             # Event Loop
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'SAVE AND EXIT'):
            write_logfile(sub_obj_list)
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
        #if event=='SAVE':
        #    write_logfile(sub_obj_list)
        if type(event) is sub_qa_info:
            event.button_set_status()
            image_ = resize_image(event.image_r,
                          resize=(int(imgsize),int(imgsize)) ,  
                          status=event.status,
                          text_val=event.subject
                          )
            window[event].update(image_data=image_) 
        if modify_frame == True:
            window.close()
            window=create_window_layout(sub_obj_list,
                                            qa_type=QAtype, grid_size=GRID_SIZE,
                                            frame_start_idx=idx,resize_xy=(imgsize,imgsize))
            modify_frame = False
    
    window.close()
    logging.info("REVIEW_FINISH")