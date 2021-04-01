#!/usr/bin/env python
# coding: utf-8


import umap
import mne, numpy as np
import seaborn as sns
import glob
import pandas as pd
import os, os.path as op

from sklearn.preprocessing import StandardScaler

from scipy.stats import kurtosis
from scipy.signal import welch

import pylab



def get_subjid(filename):
    return os.path.basename(filename).split('_')[0]

def get_raw_subj_data(subjid, topdir='/fast/ICA/*/'):
    glob_cmd = os.path.join(topdir, subjid+'*_300srate.fif')
    return glob.glob(glob_cmd)[0]

def get_distribution(filename):
    #Datasets are in the following formate /topdir/Distribution/Dataset
    return filename.split('/')[-2]

def assign_repo_EEG_labels(dframe):
    '''
    Add repo specific EEG names to data

    Parameters
    ----------
    dframe : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    dframe : pd.DataFrame
        DataFrame with EOG and ECG labels marked as veog, heog, ekg.

    '''
    dframe['veog'] = None
    dframe['heog'] = None
    dframe['ekg'] = None    
    
    dframe.loc[dframe.distribution=='CAMCAN', ['veog','heog','ekg']] = \
        'EOG061','EOG062','ECG064'
    dframe.loc[dframe.distribution=='MOUS', ['veog','heog','ekg']] = \
        'EEG058','EEG057','EEG059'
    dframe.loc[dframe.distribution=='HCP', ['veog','heog','ekg']] = \
        'eog61','eog62','eeg64'
    dframe.loc[dframe.distribution=='NIH_HV', ['veog','heog','ekg']] = \
        None, None, None 
    return dframe




def populate_dframe(topdir='/fast/ICA/', load_ica=False):
    
    dsets=glob.glob(op.join(topdir, '*/*0-ica.fif'))    
    dframe=pd.DataFrame(dsets, columns=['ica_filename'])
    
    
    dframe['distribution']=dframe['ica_filename'].apply(get_distribution)
    
    dframe['subjid']=dframe['ica_filename'].apply(get_subjid)
    
    dframe['raw_fname'] = dframe['subjid'].apply(get_raw_subj_data)
    
    dframe.distribution.value_counts()
    
    dframe = assign_repo_EEG_labels(dframe)
    
    if load_ica == False:
        return dframe
    # else:
        # return dframe, 

def get_consistent_ch_names(current_dframe):
    '''Hack to get all the same topomap dimensions'''
    ch_names=set()
    
    for index,row in current_dframe.iterrows():
        raw = mne.io.read_raw_fif(row['raw_fname'])
        ch_names=set(raw.ch_names).union(ch_names)
    
    if current_dframe.iloc[0]['distribution']=='MOUS':
        ch_names = [i for i in ch_names if i[0]=='M']    
    # elif current_dframe.iloc[0]['distribution']=='CAMCAN':
    #     ch_names = [i for i in ch_names if i[0]=='M'] 
    
    # tmp=ica.get_sources(raw, start=0, stop=100*raw.info['sfreq'])
    # freqs, _ =welch(tmp._data, fs=raw.info['sfreq'])
    return ch_names



def assess_ICA_properties(current_dframe):
    '''Loop over all datasets and return ICA metrics'''
    # full_mat=pd.DataFrame(np.zeros([comp_num*len(current_dframe), len(ch_names)]), columns=ch_names)
    current_dframe.reset_index(inplace=True)
    
    raw = mne.io.read_raw_fif(current_dframe.iloc[0]['raw_fname'])
    ch_names = get_consistent_ch_names(current_dframe)
    ica = mne.preprocessing.read_ica(current_dframe.iloc[0]['ica_filename'])
    ica_timeseries = ica.get_sources(raw, start=0, stop=100*raw.info['sfreq'])
    
    
    comp_num, samples = ica_timeseries._data.shape
    
    freqs, _ = welch(ica_timeseries._data, fs=raw.info['sfreq'])
    
    spectra_dframe = pd.DataFrame(np.zeros([comp_num*len(current_dframe), 
                                            len(freqs)]), columns = freqs)
    spectra_dframe['kurtosis'] = 0


    for index,row in current_dframe.iterrows():
        print(index)
        veog_ch, heog_ch, ekg_ch = row[['veog', 'heog', 'ekg']]
        
        ica = mne.preprocessing.read_ica(row['ica_filename'])
        component = ica.get_components()
        
        raw = mne.io.read_raw_fif(row['raw_fname']) #, preload=True)
        ch_indexs = set(raw.ch_names).intersection(ch_names)
        
    #     full_mat.loc[index*comp_num:(index*comp_num + comp_num-1), ch_indexs]=component.T
        
        raw = mne.io.read_raw_fif(row['raw_fname'], preload=True)
        
        ica_timeseries = ica.get_sources(raw, start=0, stop=100*raw.info['sfreq'])
        freqs, power = welch(ica_timeseries._data, fs=raw.info['sfreq'])
        log_power = np.log(power) 
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), freqs]=log_power
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'kurtosis'] = kurtosis(ica_timeseries._data, axis=1)
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'component_num']= range(comp_num)
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'subjid'] = row['subjid']
        
        try :
            bads_ecg=ica.find_bads_ecg(raw, ch_name=ekg_ch)[1]
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'ecg_bads_corr'] = bads_ecg
        except:
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'ecg_bads_corr'] = np.NaN
        
        try:
            bads_ecg_ctps = ica.find_bads_ecg(raw,method='ctps')[1]
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'ecg_bads_ctps'] = bads_ecg_ctps
        except:
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'ecg_bads_ctps'] = np.NaN
        
        try:
            bads_veog = ica.find_bads_eog(raw, ch_name=veog_ch)[1]
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'veog_bads_corr'] = bads_veog
        except:
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'veog_bads_corr'] = np.NaN

        try:
            bads_heog = ica.find_bads_eog(raw, ch_name=heog_ch)[1]
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'heog_bads_corr'] = bads_heog
        except:
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'heog_bads_corr'] = np.NaN            
            
            # veog_corr[index*comp_num:(index*comp_num + comp_num)] = ica.find_bads_eog(raw, ch_name=veog_ch)[1]
            # heog_corr[index*comp_num:(index*comp_num + comp_num)] = ica.find_bads_eog(raw, ch_name=heog_ch)[1]

    return spectra_dframe

# Make an input dataframe of paths
dframe = populate_dframe()

# Loop over ICA filepaths to save out csv files
for dist in dframe.distribution.unique():
    current_dframe = dframe[dframe.distribution==dist]
    out_dframe = assess_ICA_properties(current_dframe)
    out_dframe.to_csv(f'/fast/ICA/Spectra_{dist}.tsv', sep='\t', index=None)

# Compile csv files into larger dataframe again
dlist = []
for repo in ['CAMCAN', 'HCP','MOUS','NIH_HV']:
    if 'tmp' in locals().keys() : del tmp
    tmp = pd.read_csv(f'Spectra_{repo}.tsv', sep='\t')
    tmp['distribution'] = repo
    dlist.append(tmp)
combined = pd.concat(dlist)


##########################

# dframe.distribution.unique()
# dist = 'MOUS'
# current_dframe = dframe[dframe.distribution==dist]
# out_dframe = assess_ICA_properties(current_dframe)
# out_dframe.to_csv(f'/fast/ICA/Spectra_{dist}.tsv', sep='\t', index=None)

# dist = 'CAMCAN'
# current_dframe = dframe[dframe.distribution==dist]
# out_dframe = assess_ICA_properties(current_dframe)
# out_dframe.to_csv(f'/fast/ICA/Spectra_{dist}.tsv', sep='\t', index=None)

# dist = 'HCP'
# current_dframe = dframe[dframe.distribution==dist]
# out_dframe = assess_ICA_properties(current_dframe)
# out_dframe.to_csv(f'/fast/ICA/Spectra_{dist}.tsv', sep='\t', index=None)

# dist = 'NIH_HV'
# current_dframe = dframe[dframe.distribution==dist]
# out_dframe = assess_ICA_properties(current_dframe)
# out_dframe.to_csv(f'/fast/ICA/Spectra_{dist}.tsv', sep='\t', index=None)






mous_idxs = dframe[dframe.distribution=='MOUS'].index


ica = mne.preprocessing.read_ica(dframe['ica_filename'][mous_idxs[0]])
component = ica.get_components()



component.shape


dframe.columns


# chan_num, comp_num = component.shape

# #full_mat=np.zeros([comp_num*len(dframe), chan_num])
# # ctps_vec = np.zeros(comp_num*len(dframe))
# ecg_corr = np.zeros(comp_num*len(dframe))
# #eog_corr = np.zeros(comp_num*len(dframe))
# eog_061_corr = np.zeros(comp_num*len(dframe))
# eog_062_corr = np.zeros(comp_num*len(dframe))


#test=ica.find_bads_eog(raw, ch_name='EEG051') #ch_name='EOG061')


dframe.columns


current_dframe=dframe[dframe.distribution=='MOUS']
chan_num, comp_num = component.shape

full_mat=np.zeros([comp_num*len(current_dframe), chan_num])
# ctps_vec = np.zeros(comp_num*len(dframe))
ekg_corr = np.zeros(comp_num*len(current_dframe))
veog_corr = np.zeros(comp_num*len(current_dframe))
heog_corr = np.zeros(comp_num*len(current_dframe))


current_dframe.reset_index(inplace=True)

## Necessary for topomap based analysis >>>>>
# ### Get the full set of channel names

# ch_names=set()

# for index,row in current_dframe.iterrows():
#     raw = mne.io.read_raw_fif(row['raw_fname'])
#     ch_names=set(raw.ch_names).union(ch_names)
    
# ch_names = [i for i in ch_names if i[0]=='M']    

# tmp=ica.get_sources(raw, start=0, stop=100*raw.info['sfreq'])
# freqs, _ =welch(tmp._data, fs=raw.info['sfreq'])

###  <<<<<



    
    


# ## Save outputs

output_dir = '/fast/ICA/output_vars'
proj_name = 'mous'
#Save dframe as csv

full_mat.to_csv(op.join(output_dir, proj_name+'_fullmat.csv'), index=False)
spectra_dframe.to_csv(op.join(output_dir, proj_name+'_spectra_dframe.csv', index=False))

#Save numpy arrays
np.save(op.join(output_dir, proj_name+'_ekg_corr.npy'), ekg_corr)
np.save(op.join(output_dir, proj_name+'_veog_corr.npy'), veog_corr) 
np.save(op.join(output_dir, proj_name+'_heog_corr.npy'), heog_corr) 


#ctps_vec.__len__()


full_mat.shape


# # Calculate the UMAP embedding

# ### Topographic Clustering

reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.0,
                    metric='cosine') #manhattan') #sine') #'manhattan')
embedding = reducer.fit_transform(np.abs(full_mat.values))


# ### Frequency Clustering

reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.0,
                    metric='cosine') #manhattan') #sine') #'manhattan')
embedding = reducer.fit_transform(spectra_dframe.values)


# ### Combined Clustering

comb_array = np.hstack([full_mat.values, spectra_dframe.values])
reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.0,
                    metric='cosine') #manhattan') #sine') #'manhattan')
embedding = reducer.fit_transform(comb_array)


# ### Designate ECG and EOG

highecg_embedding = embedding[np.abs(ekg_corr)>.5]
higheog_embedding = embedding[(np.abs(heog_corr) > .3) | (np.abs(veog_corr) > .3)]

highveog_embedding = embedding[(np.abs(veog_corr) > .3)]
highheog_embedding = embedding[(np.abs(heog_corr) > .3)]


np.where(np.abs(heog_corr)>.3)


np.where(np.abs(veog_corr)>.3)


pylab.scatter(embedding[:,0], embedding[:,1])


fig = pylab.figure(figsize=[20,7])
pylab.plot(np.abs(veog_corr[0:400]))
pylab.plot(np.abs(heog_corr[0:400]))
#pylab.plot(ekg_corr[0:400])


fig=pylab.figure()
ax = fig.sub


#ecg_corr=np.load('./output_vars/ecg_corr.npy')


sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=np.abs(veog_corr))
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=np.abs(heog_corr))
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=np.abs(ekg_corr))


embedding[(veog_corr > .3) | (heog_corr > .3)] #or embedding[eog_62_corr > .3]


sns.scatterplot(x=embedding[:,0], y=embedding[:,1])


sns.scatterplot(x=highecg_embedding[:,0], y=highecg_embedding[:,1]) 
sns.scatterplot(x=highveog_embedding[:,0], y=highveog_embedding[:,1])
sns.scatterplot(x=highheog_embedding[:,0], y=highheog_embedding[:,1])

#sns.scatterplot(x=higheog_embedding[:,0], y=higheog_embedding[:,1])


sns.scatterplot(x=embedding[:,1], y=embedding[:,2])
sns.scatterplot(x=highecg_embedding[:,1], y=highecg_embedding[:,2]) 
#sns.scatterplot(x=higheog_embedding[:,0], y=higheog_embedding[:,1])

sns.scatterplot(x=highveog_embedding[:,1], y=highveog_embedding[:,2])
sns.scatterplot(x=highheog_embedding[:,1], y=highheog_embedding[:,2])


spectra_dframe




