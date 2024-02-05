#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:28:12 2023

@author: nugenta
"""

import os, os.path as op
import re
import sys
import mne_bids
from mne_bids import BIDSPath
import glob
import munch
import pandas as pd
import argparse

def get_subdirectories(path: str, stem: str) -> list:
    """Get a list of subdirectorty names that start with the requested stem; return strings without stem"""
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)) and item.startswith(stem):
            dirs.append(item[len(stem):])
    return dirs

def get_runnumbers(path: str, stem: str) -> list:
    """ Get a list of run numbers for a given 'task-stem' string"""
    run_numbers = set()
    for file in os.listdir(path):
        if f'task-{stem}_' in file:
            match = re.search(r'run-(\d+)', file)
            if match:
                run_numbers.add(match.group(1))
    return list(run_numbers)

def assemble_cmd(row, bids_root=None):
    '''
    Use the pandas series to generate the process_meg command

    Parameters
    ----------
    row : pd.Series
        Row from Dataframe
    bids_root : str
        DESCRIPTION. The default is None.

    Returns
    -------
    cmd_ : str
        Line in the swarmfile command.

    '''
    cmd_ = f'process_meg.py -bids_root {bids_root}'
    tag_dict = {
        'sub':'-subject',
        'ses':'-session',
        'run':'-run',
        'rest_tag':'-rest_tag',
        'eroom_tag':'-emptyroom_tag',
                }
    if len(row['eroom']) != 0 :
        row['eroom_tag']=op.basename(row.eroom).split('_task')[1].split('_')[0][1:]
    else:
        row['eroom_tag']=None
    if len(row['path']) != 0 : 
        row['rest_tag']=op.basename(row.path).split('_task')[1].split('_')[0][1:]
    else:
        raise ValueError('Could not find MEG rest dataset')
    
    for row_tag, cmd_flag in tag_dict.items():
        if (row[row_tag]==None) or (len(row[row_tag])==0):
            cmd_ += f' {cmd_flag} None'
        else:
            cmd_ += f' {cmd_flag} {row[row_tag]}'
    cmd_ += ' -n_jobs 6'
    cmd_ += '\n'
    return cmd_
    

def dframe_toswarm(dframe, bids_root=None, outfile='enigma_swarm.sh'):
    "Convert the dataframe to swarm file"
    swarm = []
    for idx, row in dframe.iterrows():
        swarm.append(assemble_cmd(row, bids_root=bids_root))
    with open(outfile, 'w') as f:
        f.writelines(swarm)
                     
        

if __name__=='__main__':

    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''The name of the BIDS directory to be parsed''')
    parser.add_argument('-rest_tag',help='''The filename stem to find rest datasets''')
    parser.add_argument('-emptyroom_tag',help='''The filename stem to find emptyroom datasets''')
    parser.add_argument('-swarmfile', 
                        help='''For internal use at NIH.  Write a swarmfile for biowulf''',
                        default=False, action='store_true')
    parser.description='''This python script parses a BIDS directory into a CSV file with one line per MEG to process'''
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(1)            
    
    if not args.bids_root:
        bids_root = 'bids_out'
    else:
        bids_root = args.bids_root
        
    if not os.path.exists(bids_root):
        raise ValueError('BIDS root directory %s does not exist' % bids_root)
                   
    if args.rest_tag:
        rest_tag = args.rest_tag
    else:
        rest_tag = 'rest'
        
    if args.emptyroom_tag:
        emptyroom_tag = args.emptyroom_tag
    else:
        emptyroom_tag = 'emptyroom'

    # define list of MEG file format types to search for       
    type_list = ['.fif','.ds','4d','.con','.sqd','rfDC']

    # initialize the BIDSPath object
    bids_path = BIDSPath(root=bids_root)

    # compile subject list and set up a dataframe to hold subject data
    print('Compiling subject list. This may take time with lots of subjects')
    #subject_list = mne_bids.get_entity_vals(bids_root, 'subject')
    
    subject_list = get_subdirectories(bids_root, 'sub-')
    
    allsubj_df = pd.DataFrame(columns=['sub','ses','run','type','path','mripath','eroom'], 
                              dtype=str)

    for subject in subject_list:
    
        print('Working on subject %s' % subject)
        subj_dir = os.path.join(bids_root,'sub-' + subject)
    
        # initialize subject lists
        subrestlist = []
        mrilist = []
    
        rest_count = 0
        mri_count = 0
        
        # to exclude all the other subjects when searching for datasets from this subject, we'll
        # need to have a list of all other subjects besides the present one
        #other_subjects = list(set(subject_list) - set([subject]))

        # list of sessions for this subject
        #session_list = mne_bids.get_entity_vals(bids_root, 'session', ignore_subjects = other_subjects)
        
        session_list = get_subdirectories(subj_dir, 'ses-')
        print('%d sessions for subject %s' % (len(session_list),subject))
        
        # first, we are going to make a list of all available anatomical MRI scans
        
        for session in session_list:
            
            anatpath = bids_root + '/sub-' + subject + '/ses-' + session + '/anat/'
            mrifiles = glob.glob(f'{anatpath}*T1w.nii*')
                
            if len(mrifiles) > 0:   # if mri files found in current session, add to mri object list
                
                mri_count = mri_count+1
                mri_object = munch.Munch()
                mri_object.ses = session
                mri_object.path = mrifiles[0]
                mrilist.append(mri_object)
           
            if len(mrifiles) > 1:   # if there's more than one MRI in a session, just pick the first one
                print("More than one MRI in session %s, picking first one" % session)
                
        print("Identified %d T1w MRI scans for subject %s" % (len(mrilist), subject))
                
        # now, we are going to make a list of all the resting state MEG scans
        
        for session in session_list:   
            
            subrestlist_ses = []
            
            session_dir = os.path.join(subj_dir,'ses-' + session)
            # only look at the current session, that requires a list of all the other sessions
            #other_sessions = list(set(session_list) - set([session]))
            #run_list = mne_bids.get_entity_vals(bids_root, 'run', ignore_subjects = other_subjects,
            #                    ignore_sessions=other_sessions)
            
            megpath = bids_root + '/sub-' + subject + '/ses-' + session + '/meg'
    
            # is there an MEG in the session?
            if(os.path.exists(megpath)):
        
                print('found a megpath')
                # make a list of the runs, and make a new entry for each run
                run_list = get_runnumbers(megpath, rest_tag)
                print('found %d rest runs for session %s' % (len(run_list), session))
                for run in run_list:
                
                    # we'll have to search for available MEGs looping over all datatypes
                    for dattype in type_list:
                        restfiles = glob.glob(f'{megpath}/*{rest_tag}*run-{run}*{dattype}')
                        
                        # if an MEG dataset is found, create an object 
                        if len(restfiles) > 0:
                            print('found an MEG')
                            record_count = rest_count + 1
                            proc_object = munch.Munch()
                            proc_object.sub = subject
                            proc_object.ses = session
                            proc_object.run = run
                            proc_object.type = dattype
                            proc_object.path = restfiles[0]
                            proc_object.mripath = ''
                            proc_object.eroom = ''
                            subrestlist_ses.append(proc_object)
                
                        
                # look for emptyroom datasets in the same session, don't care about run
                for dattype in type_list:
                    emptyfiles = glob.glob(f'{megpath}/*{emptyroom_tag}*{dattype}')
                
                    if len(emptyfiles) > 0:
                        print('found an emptyroom')
                        for subrest_ses in subrestlist_ses:     
                            subrest_ses.eroom = emptyfiles[0]
            subrestlist.extend(subrestlist_ses)
       
        # OKAY. So at this point we have two lists. One has all the MRIs, and one has all the 
        # MpdEGs. We have to match them up. 

        if len(mrilist) != 0:
            for restmeg in subrestlist:

                print('going through list of MEGs for subject %s and matching with MRI' % subject)
                # look at the session of the MEG and see if there is an MRI in the same session
                for mri in mrilist:                
                    if mri.ses == restmeg.ses:                    
                        restmeg.mripath = mri.path
                        print('Session %s match for MEG and MRI' % mri.ses)
              
                # what happens if there is no mri in the same sesssion as the meg           
                if restmeg.mripath == '':
                    restmeg.mripath = mrilist[0].path
                    print('No MRI MEG session match, using an MRI from same subject, different session')
            
            # now, merge all the proc objects into a single dictionary, and make that a dataframe. 
        else:
            print('No MRI for this participant')
    
        # once we've finished with the subject, unmunchify the list of objects and make a dataframe for the subject
        subrestlist = munch.unmunchify(subrestlist)     
        subj_df = pd.DataFrame(subrestlist)
    
        # concatenate the subject dataframe with the global datafram for all subjects
        allsubj_df = pd.concat([allsubj_df, subj_df])

    if args.swarmfile == True:
        dframe_toswarm(allsubj_df, bids_root=bids_root,
                       outfile=op.join(os.getcwd(),'enigma_swarmfile.sh'))
    else:
        # save out the dataframe as a .csv file
        allsubj_df.to_csv('ParsedBIDS_dataframe.csv', index=False)
    
