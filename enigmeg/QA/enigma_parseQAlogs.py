#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:46:31 2023

@author: Allison Nugent and chatGPT
"""

import os.path as op
import argparse
import glob
import pandas as pd
from enigmeg.QA.enigma_QA_GUI_functions import initialize, get_last_review, get_subject_status

PROJECT='ENIGMA_MEG_QA'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''Location of bids directory, default=bids_out''')
    
    args = parser.parse_args()

    if not args.bids_root:
        bids_root='bids_out'
    else:
        bids_root=args.bids_root
    if not op.exists(bids_root):
        parser.print_help()
        raise ValueError('specified BIDS directory %s does not exist' % bids_root)
        
    deriv_root = op.join(bids_root, 'derivatives') 
    QA_root = op.join(deriv_root,PROJECT)
    
    history_log = []
    
    QA_list = glob.glob(op.join(QA_root,'*_QA_logfile.txt'))
    
    for QA_log in QA_list:
        
        QAtype = QA_log.split('/')[-1].split('_')[0]
        print('Parsing log for %s QA now' % QAtype)
        log = initialize(bids_root, QAtype)
        last = get_last_review(log)
        if last != []:
            history_log.append(last)
            
        # flatten the history list, thanks chatGPT
        
        history_log = [item for sublist in history_log for item in sublist]
            
    data_dict = {}        
            
    for row in history_log:
        
        subject, status, qatype = get_subject_status(row)
        
        if subject not in data_dict:
            data_dict[subject] = {}
        data_dict[subject][qatype] = status
        
    data_frame = pd.DataFrame.from_dict(data_dict, orient='index')    
    
    dataframe_path = op.join(QA_root,'QA_summary.csv')
    data_frame.to_csv(dataframe_path)
    