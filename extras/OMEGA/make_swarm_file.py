#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:45:53 2023

@author: jstout
"""
import pandas as pd
import os, os.path as op
import glob

bids_dir = '/data/EnigmaMeg/BIDS/Omega'
os.chdir(bids_dir)

#fname = 'OMEGA_rest_sets.csv'
#dframe = pd.read_csv(fname)
inputs=glob.glob('sub-*/ses-*/meg/*rest*.ds')
dframe = pd.DataFrame(inputs, columns=['rest_fname'])

def get_subjid(fname):
    return fname.split('/')[0]

def get_session(fname):
    return fname.split('/')[1].split('-')[1]

def get_task(fname):
    try:
        return fname.split('task-')[1].split('_')[0]
    except:
        return False

def get_run(fname):
    return fname.split('run-')[1].split('_')[0]

dframe['subjid']=dframe.rest_fname.apply(get_subjid)    
dframe['session']=dframe.rest_fname.apply(get_session)
dframe['task'] = dframe.rest_fname.apply(get_task)
dframe['run'] = dframe.rest_fname.apply(get_run)

dframe.drop(dframe[dframe.task!='rest'].index, inplace=True)

dframe.sort_values(by=['subjid','task','run'])
dframe.drop_duplicates(subset='subjid',
                       keep='first',
                       inplace=True)

swarm_list = []
for i, row in dframe.iterrows():
    outmsg=f'process_meg.py -bids_root {bids_dir} -subject {row.subjid[4:]} -session {row.session} -run \
          {row.run} -emptyroom_tag noise -rest_tag rest  -fs_ave_fids -mains 60.0'
    outmsg=' '.join(outmsg.split())+'\n'
    swarm_list.append(outmsg)

swarm_fname = f'{bids_dir}/swarm_omega_enigma.sh'
with open(swarm_fname, 'w+') as f:
    f.writelines(swarm_list)
    

