#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:45:53 2023

@author: jstout
"""
import pandas as pd
import os, os.path as op
cd ~

#glob.glob('sub-*/ses-*/meg/*rest*.ds')
fname = 'OMEGA_rest_sets.csv'
dframe = pd.read_csv(fname)

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


# session_pick = dframe.groupby('subjid')['session'].min().reset_index()

# def return_final(row):
#     return dframe.loc[(dframe.subjid==row.subjid) & (dframe.session==row.session)]
    

# for i,row in session_pick.iterrows():
#     print(return_final(row))



