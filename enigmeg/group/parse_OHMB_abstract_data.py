#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:41:33 2023

@author: jstout
"""

import os, os.path as op
import pandas as pd
import glob
import mne
from mne.viz import Brain
import nibabel as nib
import numpy as np
import seaborn as sns
import pylab
import nibabel as nib

import statsmodels.api as sm
import statsmodels.formula.api as smf


topdir = '/home/jstout/src/enigma_OHBM/RESULTS_OHBM'
os.chdir(topdir)

def return_multisubject_dframe(filenames):
    '''Take a list of filenames 
    Load the csv filenames into a dataframe
    Stitch together single subject dataframes into one large dataframe'''
    
    list_of_dframes = list()
    for fname in filenames:
        print(fname)
        subj_dframe = pd.read_csv(fname, sep='\t')       
        subj_dframe['subject'] = fname.split('/')[1].split('_')[0]
        subj_dframe['group'] = fname.split('/')[0]
        list_of_dframes.append(subj_dframe)
        
    group_dframe = pd.concat(list_of_dframes)
    group_dframe = group_dframe.rename(columns={'Unnamed: 0':'Parcel'})
    return group_dframe
 

groups = ['MEGUK_card',
     'CAMCAN',
     'MOUS',
     'OMEGA',
     'NIH_hv',
     'NIH_string']


group_var=[]
fnames=[]
for group in groups:
    tmp=glob.glob(f'{group}/*_Band_rel_power.csv')
    fnames+= tmp
    group_var += [group] * len(tmp)
    
dframe = pd.DataFrame(zip(group_var, fnames), columns=['group','fnames'])
group = return_multisubject_dframe(fnames)
group.reset_index(drop=True, inplace=True) #keep=False)
ave_cats= ['[1, 3]', '[3, 6]', '[8, 12]', '[13, 35]', '[35, 55]',
       'AlphaPeak']

stats_dframe=group.groupby('Parcel')[ave_cats].mean()


# =============================================================================
# 
# =============================================================================
def filter_parcel_data(dframe, parcel_name=None, column_name=None):
    '''Return single parcel information for stats testing'''
    tmp=dframe[dframe.Parcel==parcel_name]
    if ('age' in dframe.columns) and ('sex' in dframe.columns):
        return tmp[[column_name, 'age', 'sex']].dropna()
    else:
        return tmp[[column_name]].dropna()

def proc_parcel_regression(dframe):
    '''Takes merged dataframe (subj_vars + parcel data)
    and performs regression of variables using statsmodels'''
    
    #Initialize regression dataframe
    rdframe = pd.DataFrame(columns=['parcel_name', 'coeff','rsquared_adj'])
    rdframe.parcel_name = dframe.Parcel.unique()

    for idx, row in rdframe.iterrows():
        parc = row['parcel_name']
        tmp=filter_parcel_data(dframe, parcel_name=parc, 
                               column_name='AlphaPeak')
        results = smf.ols('AlphaPeak ~ age', data=tmp).fit()
        rdframe.loc[idx, 'rsquared_adj']=results.rsquared_adj
        rdframe.loc[idx, 'coeff']=results.params['age']
    
    return rdframe


def display_regression_coefs(stats_dframe, 
                             subject_id='fsaverage',
                             parc_name='aparc_sub',
                             subjects_dir=None,
                             image_outpath=None, 
                             col_id=None, 
                             rel_scaling=False, 
                             fmin=None,
                             fmax=None):
    '''Plot the statistics on the brain
    parc_name:
        aparc or aparc_sub currently supported
    
    '''
   
    if subjects_dir != None: os.environ['SUBJECTS_DIR']=subjects_dir
    hemi = "lh"
    surf = "inflated"
    
    brain = Brain(subject_id, hemi, surf, background="white")
    
    aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                              subject_id, "label",
                              hemi + f".{parc_name}.annot") 
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
    
    names2=[i.decode() for i in names] #convert from binary
    if 'corpuscallosum' in names2: names2.remove('corpuscallosum')
    if 'unknown' in names2: names2.remove('unknown')
    
    # Placeholder - must be larger than number of ROIs
    roi_data = np.zeros(600)
    roi_data[:] = np.nan
    
    for idx,name in enumerate(names2):
        roi_data[idx]=stats_dframe.loc[name+'-'+hemi, col_id]
    	
    vtx_data = roi_data[labels]
    vtx_data[labels == -1] = 0
    
    thresh=.001
    vtx_data[np.abs(vtx_data)<thresh]=0
    fmin = stats_dframe[col_id].min()
    fmax = stats_dframe[col_id].max()
    brain.add_data(vtx_data, colormap="jet", fmin=fmin, fmax=fmax)
    
    if image_outpath != None:
        brain.save_image(image_outpath)

# =============================================================================
# Average images
# =============================================================================
#Alpha Peak
display_regression_coefs(stats_dframe, col_id='AlphaPeak') 

#Delta
display_regression_coefs(stats_dframe, col_id='[1, 3]') 

#Theta
display_regression_coefs(stats_dframe, col_id='[3, 6]')

#Alpha
display_regression_coefs(stats_dframe, col_id='[8, 12]')

#Beta
display_regression_coefs(stats_dframe, col_id='[13, 35]')

#Low Gamma
display_regression_coefs(stats_dframe, col_id='[35, 55]')

# =============================================================================
# Regressor Images
# =============================================================================
#Load and fix names before merge
demo_fname='/home/jstout/src/enigma_OHBM/RESULTS_OHBM/demographics.csv'
demo_dframe = pd.read_csv(demo_fname)

rename_dict=dict(NIHhv='NIH_hv',
                 NIHstr='NIH_string')
demo_dframe.Site.replace(rename_dict, inplace=True)

#Merge
final = pd.merge(demo_dframe, group, left_on=['Site','participant_id'], right_on=['group','subject'])
#Required renaming because stats models does not accept numbers as names
col_rename={'[1, 3]':'Delta',
            '[3, 6]':'Theta',
            '[8, 12]':'Alpha', 
            '[13, 35]':'Beta',
            '[35, 55]': 'L_Gamma'}
final.rename(columns=col_rename, inplace=True)

# Make regressors

def proc_parcel_regression(dframe, col_id='AlphaPeak'):
    '''Takes merged dataframe (subj_vars + parcel data)
    and performs regression of variables using statsmodels'''
    
    #Initialize regression dataframe
    rdframe = pd.DataFrame(columns=['Parcel', 'coeff','rsquared_adj'])
    rdframe.Parcel = dframe.Parcel.unique()

    for idx, row in rdframe.iterrows():
        parc = row['Parcel'] #'parcel_name']
        tmp=filter_parcel_data(dframe, parcel_name=parc, 
                               column_name=col_id)
        results = smf.ols(f'{col_id} ~ age', data=tmp).fit()
        rdframe.loc[idx, 'rsquared_adj']=results.rsquared_adj
        rdframe.loc[idx, 'coeff']=results.params['age']
    rdframe.set_index('Parcel', inplace=True)
    return rdframe


#AlphaPeak
rdframe=proc_parcel_regression(final.dropna(subset=['AlphaPeak']))
display_regression_coefs(rdframe, col_id='rsquared_adj')

#Delta
rdframe=proc_parcel_regression(final, col_id='Delta')
display_regression_coefs(rdframe, col_id='rsquared_adj')

#Theta
rdframe=proc_parcel_regression(final, col_id='Theta')
display_regression_coefs(rdframe, col_id='rsquared_adj')

#Alpha
rdframe=proc_parcel_regression(final, col_id='Alpha')
display_regression_coefs(rdframe, col_id='rsquared_adj')

#Beta
rdframe=proc_parcel_regression(final, col_id='Beta')
display_regression_coefs(rdframe, col_id='rsquared_adj')

#L_Gamma
rdframe=proc_parcel_regression(final, col_id='L_Gamma')
display_regression_coefs(rdframe, col_id='rsquared_adj')

# =============================================================================
# Final number counts
# =============================================================================
test = final.groupby(['group']).subject.apply(set)
for group, subject in test.iteritems():
    print(f'{group} has {len(subject)}')
    
final.groupby(['group']).age.agg([np.mean, np.median, np.min, np.max])

test2=final.groupby(['group', 'subject']).first()
test2.reset_index(inplace=True)



