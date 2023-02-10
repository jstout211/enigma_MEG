#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:07:24 2023

@author: jstout
"""
from enigmeg.QA.ImageSelectorGui import get_last_review, get_subject_status, build_status_dict
import os.path as op
logfile = op.join(op.dirname(__file__) , 'enigma_QA_logfile.txt')

with open(logfile) as f:
    history_log = f.readlines()
#Strip newlines        
history_log=[i[:-1] for i in history_log if i[-1:]=='\n']

last_review = get_last_review(history_log) 

def test_get_last_review():
    last_review = get_last_review(history_log)
    assert len(last_review)==49  

def test_get_subject_status():
    subj,status = get_subject_status(last_review[2])
    assert (subj=='24235') & (status=='Unchecked')
    subj,status = get_subject_status(last_review[3])
    assert (subj=='23693') & (status=='GOOD')

def test_build_status_dict():
    tmp_ = build_status_dict(last_review)
    gtruth = ['24208', '24216', '24235', '23693', '24252', '24259', '23607', '24085', '24295', '22695', '24232', '24263', '24227', '23593', '24185', '24138', '24267', '23732', '23641', '24175', '22694', '21111', '23550', '23540', '22812', '24225', '24213', '24169', '24229', '23757', '24103', '24172', '23656', '24287', '23780', '24281', '23490', '23809', '23777', '24199', '23672', '24071', '23927', '23911', '24286', '24201', '23951', '23520', '24238']
    assert list(tmp_.keys()) == gtruth
    gtruth = {'24208': 'Unchecked',
             '24216': 'Unchecked',
             '24235': 'Unchecked',
             '23693': 'GOOD',
             '24252': 'BAD',
             '24259': 'GOOD'}
    #Find the subjects and assert that their status is the same in both 
    for subj, status in gtruth.items():
        assert tmp_[subj]==status




