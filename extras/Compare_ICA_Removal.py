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
        'EOG061','EOG062','ECG063'
    dframe.loc[dframe.distribution=='MOUS', ['veog','heog','ekg']] = \
        'EEG058','EEG057','EEG059'
    dframe.loc[dframe.distribution=='HCP', ['veog','heog','ekg']] = \
        'VEOG','HEOG','ECG'
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

def calc_hcp_bipolar(row):
    '''Load info from the mne-hcp and return bipolar calculated ECG, VEOG, HEOG'''
    subjid = row.subjid
    
    #info read from mne-hcp not the same as the one tied to the raw dataset
    info = mne.io.read_info(f'/fast/ICA/HCPinfo/{subjid}-info.fif')
    
    raw=mne.io.read_raw_fif(row.raw_fname, preload=True)
    ecgPos_idx=info.ch_names.index('ECG+')
    ecgNeg_idx=info.ch_names.index('ECG-')
    veogPos_idx=info.ch_names.index('VEOG+')
    veogNeg_idx=info.ch_names.index('VEOG-')
    heogPos_idx=info.ch_names.index('HEOG+')
    heogNeg_idx=info.ch_names.index('HEOG-')
    
    ecg=raw._data[ecgPos_idx,:]-raw._data[ecgNeg_idx,:]
    veog=raw._data[veogPos_idx,:]-raw._data[veogNeg_idx,:]
    heog=raw._data[heogPos_idx,:]-raw._data[heogNeg_idx,:]
    
    raw._data[ecgPos_idx,:]=ecg
    raw._data[veogPos_idx,:]=veog
    raw._data[heogPos_idx,:]=heog
    
    raw.rename_channels({raw.ch_names[ecgPos_idx]:'ECG'})
    raw.rename_channels({raw.ch_names[veogPos_idx]:'VEOG'})
    raw.rename_channels({raw.ch_names[heogPos_idx]:'HEOG'})
    
    raw.drop_channels(raw.ch_names[ecgNeg_idx])
    raw.drop_channels(raw.ch_names[veogNeg_idx])
    raw.drop_channels(raw.ch_names[heogNeg_idx])
    
    return raw

def assess_ICA_spectral_properties(current_dframe):
    '''Loop over all datasets and return ICA metrics'''
    current_dframe.reset_index(inplace=True)
    
    #Load first dataset to allocate size to the dataframe
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
        
        if row.distribution == 'HCP':
            raw = calc_hcp_bipolar(row)
        else:
            raw = mne.io.read_raw_fif(row['raw_fname'], preload=True)
        
        ch_indexs = set(raw.ch_names).intersection(ch_names)
        
        
        # raw = mne.io.read_raw_fif(row['raw_fname'], preload=True)
        
        ica_timeseries = ica.get_sources(raw, start=0, stop=100*raw.info['sfreq'])
        freqs, power = welch(ica_timeseries._data, fs=raw.info['sfreq'])
        log_power = np.log(power) 
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), freqs]=log_power
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'kurtosis'] = kurtosis(ica_timeseries._data, axis=1)
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'component_num']= range(comp_num)
        spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'subjid'] = row['subjid']
        
        try :
            bads_ecg=ica.find_bads_ecg(raw, ch_name=ekg_ch, method='correlation')[1]
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'ecg_bads_corr'] = bads_ecg
        except:
            spectra_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'ecg_bads_corr'] = np.NaN
        
        try:
            bads_ecg_ctps = ica.find_bads_ecg(raw, ch_name=ekg_ch, method='ctps')[1]
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

def plot_topo_hack(normalized_topo):
    ref_sens = mne.io.read_raw_fif('/fast/ICA/CAMCAN/sub-CC621184_ses-rest_task-rest_proc-sss_300srate.fif')
    ref_sens.crop(0, 2)
    ref_sens.pick_types(meg='mag')
    ref_sens.load_data()    
    epochs = mne.make_fixed_length_epochs(ref_sens)    
    evoked = epochs.average()
    if normalized_topo.shape.__len__() == 1:
         evoked._data[:,0]=normalized_topo
         evoked.plot_topomap(times=evoked.times[0], colorbar=False)
    else:
        evoked._data[:,:25]=normalized_topo
        evoked.plot_topomap(times=evoked.times[0:25], ncols=5, nrows=5, colorbar=False)


def assess_ICA_topographic_properties(current_dframe):
    '''Loop over all datasets and return ICA metrics'''
    
    ref_sens = mne.io.read_raw_fif('/fast/ICA/CAMCAN/sub-CC621184_ses-rest_task-rest_proc-sss_300srate.fif', preload=True)
    ref_sens.pick_types(meg='mag')
    
    ref_ica_fname = '/fast/ICA/CAMCAN/sub-CC621184_ses-rest_task-rest_proc-sss_0-ica.fif'
    ref_ica = mne.preprocessing.read_ica(ref_ica_fname)
    
    
    current_dframe.reset_index(inplace=True, drop=True)
    
    ica = mne.preprocessing.read_ica(current_dframe.iloc[0]['ica_filename'])
    
    _, comp_num = ica.get_components().shape #_timeseries._data.shape
    
    topo_dframe = pd.DataFrame(np.zeros([comp_num*len(current_dframe), 102]), columns = range(102))


    for index,row in current_dframe.iterrows():
        print(index)
        veog_ch, heog_ch, ekg_ch = row[['veog', 'heog', 'ekg']]
        
        ica = mne.preprocessing.read_ica(row['ica_filename'])
        component = ica.get_components()
        
        convert_to_ref=mne.forward._map_meg_or_eeg_channels(ica.info,
                                                            ref_sens.info,
                                 # reference_sens.info, 
                                 'accurate',
                                 (0., 0., 0.04))
        
        normalized_topo = convert_to_ref @ component
        mins_= normalized_topo.min(axis=0)
        maxs_ = normalized_topo.max(axis=0)
        standardized_topo = 2 * (normalized_topo - mins_ ) / (maxs_ - mins_) - 1 
        
        topo_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), range(102)]=standardized_topo.T
        topo_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'component_num']= range(comp_num)
        topo_dframe.loc[index*comp_num:(index*comp_num + comp_num-1), 'subjid'] = row['subjid']
    return topo_dframe







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
combined.reset_index(inplace=True)

combined['ecg_bad']=combined['ecg_bads_ctps'] > 0.2
combined['eog_bad']= (np.abs(combined.heog_bads_corr) > .25) | (np.abs(combined.veog_bads_corr) > .25)



def merge_dframes_topo_spectral():
    spectral_dframe =combined[combined.distribution.isin(['CAMCAN','MOUS'])].copy()
    spectral_1_40 = spectral_dframe[spectral_dframe.columns[1:40]].copy()
    mins_= spectral_1_40.min(axis=1).values #spectral_dframemalized_topo.min(axis=0)
    maxs_ = spectral_1_40.max(axis=1).values # normalized_topo.max(axis=0)
    # standardized_spetra = 2 * (spectral_1_40 - mins_ ) / (maxs_ - mins_) - 1 
    denom = maxs_ - mins_
    numer = spectral_1_40.values - mins_[:,np.newaxis]
    standardized_spectra = 2 * numer/denom[:,np.newaxis] - 1
    spectral_dframe.iloc[:,1:40]=standardized_spectra
    from scipy.stats import zscore 
    spectral_dframe.loc[spectral_dframe['kurtosis']>50,'kurtosis']=50
    spectral_dframe.loc[:,'kurtosis']=zscore(spectral_dframe.loc[:,'kurtosis']).astype(np.float16)
    
    
    
    bads_info = spectral_dframe[['subjid', 'ecg_bad','eog_bad','component_num','distribution','kurtosis']]
    bads_info = spectral_dframe[['subjid', 'ecg_bad','eog_bad','component_num','distribution','kurtosis']\
                                + list(spectral_dframe.columns[1:40])]
    
    topo_dframe = pd.read_csv('Topo_Dframe_ELEK102_MOUS_CAM.tsv', sep='\t')
    
    dframe=pd.merge(topo_dframe, bads_info, on=['subjid','component_num'])
    dframe['bads'] = None# 'Good'
    dframe.loc[dframe['ecg_bad'],'bads']='ECG'
    dframe.loc[dframe['eog_bad'],'bads']='EOG'
    
    dframe = dframe.sample(frac=1).reset_index(drop=True)
    
    
    full_mat = pd.concat([dframe.iloc[:,range(102)],dframe.loc[:,spectral_dframe.columns[1:40]],
                          dframe.loc[:,'kurtosis']],axis=1)
    
    reducer = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.05,
                    metric='manhattan')#'cosine')#'manhattan')#'cosine') #manhattan') #sine') #'manhattan')
    embedding = reducer.fit_transform(full_mat.values) #normalized_data)
    # umap.plot(reducer, labels=dframe['bads'])
    
    fig, axes = matplotlib.pyplot.subplots(2,1, sharex=True, sharey=True,
                                           figsize=(10,10))
    
    ###  Up to the above - works
    
    
    #fig.suptitle(dist)
    
    sns.scatterplot(ax=axes[0], x=embedding[:,0], y=embedding[:,1], 
                    hue=dframe['bads'])#, style=dframe['distribution']) #np.abs(combined_dframe['ecg_bad']))
    sns.scatterplot(ax=axes[1], x=embedding[:,1], y=embedding[:,2], 
                    hue=dframe['bads'])#, style=dframe['distribu

#    sns.scatterplot(ax=axes[0,1], x=embedding[:,1], y=embedding[:,2], 
#                    hue=dframe['ecg_bad']) #np.abs(combined_dframe['ecg_bad']))

    sns.scatterplot(ax=axes[1], x=embedding[:,0], y=embedding[:,1], 
                    hue=dframe['eog_bad'])#, style=dframe['distribution'])
#    sns.scatterplot(ax=axes[1,1], x=embedding[:,1], y=embedding[:,2], 
#                    hue=dframe['eog_bad'])

    fig, axes = matplotlib.pyplot.subplots(2,2, sharex=True, sharey=True,
                                           figsize=(20,20))
    #fig.suptitle(dist)
    
    sns.scatterplot(ax=axes[0,0], x=embedding[:,0], y=embedding[:,1], 
                    hue=dframe['distribution'])#, style=dframe['distribution']) #np.abs(combined_dframe['ecg_bad']))
#    sns.scatterplot(ax=axes[0,1], x=embedding[:,1], y=embedding[:,2], 
#                    hue=dframe['ecg_bad']) #np.abs(combined_dframe['ecg_bad']))

    sns.scatterplot(ax=axes[1,0], x=embedding[:,0], y=embedding[:,1], 
                    hue=dframe['eog_bad'])#, style=dframe['distribution'])
    
    


# # Calculate the UMAP embedding

# ### Topographic Clustering

reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.0,
                    metric='cosine') #manhattan') #sine') #'manhattan')
embedding = reducer.fit_transform(np.abs(full_mat.values))


# ### Frequency Clustering

reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.0,
                    metric='cosine') #manhattan') #sine') #'manhattan')
embedding = reducer.fit_transform(spectra_dframe.values)


def plot_spectra_distr_ave(combined):
    '''Plot average spectra for different open source repositories'''
    freq_idxs = list(combined.columns[range(1,130)])
    col_idxs = freq_idxs + ['subjid', 'distribution','ecg_bad', 'eog_bad']
    tmp_dframe = combined.loc[:, col_idxs]
    melted = pd.melt(tmp_dframe, id_vars=['distribution','subjid', 'ecg_bad', 'eog_bad'] ,
                     value_vars=freq_idxs)
    sns.lineplot(x='variable', y='value', hue='distribution', data=melted)
    
def plot_ecg_std(combined):
    combined=combined[combined.distribution != 'NIH_HV']
    freq_idxs = list(combined.columns[range(1,40)])
    # freq_idxs = list(combined.columns[range(1,130)])
    col_idxs = freq_idxs + ['subjid', 'distribution','ecg_bad']
    tmp_dframe = combined.loc[:, col_idxs]
    melted = pd.melt(tmp_dframe, id_vars=['distribution','subjid', 'ecg_bad'] , value_vars=freq_idxs)
    # sns.lineplot(x='variable', y='value', hue='distribution', style='ecg_bad', data=melted)
    sns.lineplot(x='variable', y='value', hue='distribution', style='ecg_bad', data=melted, ci=np.var)

def topo_correlation_flip(reduced_dframe, ref_topo='first_val'):
    if type(ref_topo)==str: #'first_val':
        if ref_topo=='first_val':
            ref_topo=reduced_dframe.iloc[0,range(102)]
        elif ref_topo=='mean':
            ref_topo=reduced_dframe.iloc[:,range(102)].mean(axis=0)
    #If ref_topo is an array, this will be used as is

    topos=reduced_dframe.copy().iloc[:,range(102)]
    
    corrvals = [np.correlate(topos.iloc[i], ref_topo)[0] for i in range(len(topos))]
    corrvals = np.array(corrvals)
    corrvals[corrvals<0]=-1
    corrvals[corrvals>0]=1
    
    assert len(corrvals) == len(reduced_dframe)
    reduced_dframe.iloc[:,range(102)] *= corrvals[:,np.newaxis]
    
    return reduced_dframe


def plot_ecg_distribution():
    topo_dframe = pd.read_csv('Topo_Dframe_ELEK102_MOUS_CAM.tsv', sep='\t')
    
    spectral_dframe = combined[combined.distribution.isin(['CAMCAN','MOUS'])].copy()
    
    bads_info = spectral_dframe[['subjid', 'ecg_bad','eog_bad','component_num','distribution','kurtosis']]
    
    
    dframe=pd.merge(topo_dframe, bads_info, on=['subjid','component_num'])
    dframe['bads'] = 'Good'
    dframe.loc[dframe['ecg_bad'],'bads']='ECG'
    dframe.loc[dframe['eog_bad'],'bads']='EOG'
    
    topo_idxs = [str(i) for i in range(102)]
    # plot_topo_hack(dframe.loc[2,topo_idxs])
    
    eogs = dframe[dframe['bads']=='EOG'].copy()
    ecgs = dframe[dframe['bads']=='ECG'].copy()
    
    #Flip the topography based on maximal correlation to reference
    eogs = topo_correlation_flip(eogs, ref_topo='first_val')
    ecgs = topo_correlation_flip(ecgs, ref_topo='first_val')
    mean_eog = eogs.iloc[:,range(102)].mean()
    mean_ecg = ecgs.iloc[:,range(102)].mean()
    
    mean_ecg.to_csv('./ecg_average_mous_camcan.tsv', sep='\t', index=False)
    mean_eog.to_csv('./eog_average_mous_camcan.tsv', sep='\t', index=False)
    
    # eogs = topo_correlation_flip(eogs)
    # ecgs = topo_correlation_flip(ecgs)
    
    
    plot_topo_hack(ecgs.iloc[:,range(102)].mean())
    plot_topo_hack(eogs.iloc[:,range(102)].mean())
    
    plot_topo_hack(ecgs.iloc[:,range(102)].var())
    plot_topo_hack(eogs.iloc[:,range(102)].var())
    

    

    
    


g = sns.FacetGrid(melted, col="distribution", col_wrap=2, margin_titles=True)
g.map(sns.lineplot, 'variable', 'value', 'ecg_bad', ci='sd')
g.add_legend()

g = sns.FacetGrid(melted, col="distribution", col_wrap=2, margin_titles=True)
g.map(sns.lineplot, 'variable', 'value', 'eog_bad', ci='sd')
g.add_legend()

g.savefig('/home/jstout/unormalized_group2.png')    
    

def plot_kurtosis_boxplot(combined):
    sns.boxplot(x='distribution', y='kurtosis', hue='ecg_bad', data=combined,
                  dodge=True)
    
import copy
def display_umap(dframe, dist=None): #, suptitle=None):
    combined_dframe = copy.deepcopy(dframe)
    combined_dframe = combined_dframe[combined_dframe.distribution==dist]
    combined_dframe.reset_index(inplace=True, drop=True)
    
    spectra_dframe = pd.concat([combined_dframe.iloc[:,1:50],
                                combined_dframe.iloc[:,130]],
                               axis=1)
    normalized_data = StandardScaler().fit_transform(spectra_dframe)
   # # spectra_dframe = combined_dframe.iloc[:,1:50]+
   #  reducer = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0,
   #                  metric='manhattan', low_memory=False, densmap=True,
   #                  dens_lambda=0.1) #sine') #'manhattan')
    # reducer = umap.UMAP(n_components=3, n_neighbors=5, min_dist=0.5,
    #                 metric='manhattan', low_memory=False)    
    reducer = umap.UMAP(n_components=2)
    
    # embedding = reducer.fit_transform(spectra_dframe.values)
    embedding = reducer.fit_transform(spectra_dframe) #normalized_data)
    
    fig, axes = matplotlib.pyplot.subplots(2,2, sharex=True, sharey=True,
                                           figsize=(8,8))
    #fig.suptitle(dist)
    
    sns.scatterplot(ax=axes[0,0], x=embedding[:,0], y=embedding[:,1], 
                    hue=np.abs(combined_dframe['ecg_bad']))
    sns.scatterplot(ax=axes[0,1], x=embedding[:,1], y=embedding[:,2], 
                    hue=np.abs(combined_dframe['ecg_bad']))

    sns.scatterplot(ax=axes[1,0], x=embedding[:,0], y=embedding[:,1], 
                    hue=np.abs(combined_dframe['eog_bad']))
    sns.scatterplot(ax=axes[1,1], x=embedding[:,1], y=embedding[:,2], 
                    hue=np.abs(combined_dframe['eog_bad']))    
    
    
    
cmb.dropna(subset=['veog_bads_corr', 'heog_bads_corr']) #, 'ecg_bads_corr'])
combined['high_ctps'] = combined['ecg_bads_ctps'] > .3
embedding[np.abs(combined['ecg_bads_ctps'])>.3]

# fig = matplotlib.pyplot.figure()
fig, axes = matplotlib.pyplot.subplots(1,2) #fig.subplots(1,2)     
sns.scatterplot(ax=axes[0], x=embedding[:,0], y=embedding[:,1], 
                    hue=np.abs(combined_dframe['ecg_bads_corr']))
sns.scatterplot(ax=axes[1], x=embedding[:,1], y=embedding[:,2], 
                    hue=np.abs(combined_dframe['ecg_bads_corr']))

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

def test_ica_extraction():
    topdir = '/fast/ICA'
    raw_fname = op.join(topdir, 
                        'CAMCAN',
                        'sub-CC723395_ses-rest_task-rest_proc-sss_300srate.fif')
    ica_fname = op.join(topdir, 
                        'CAMCAN', 
                        'sub-CC723395_ses-rest_task-rest_proc-sss_0-ica.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    ica = mne.preprocessing.read_ica(ica_fname)
    
def load_raw_and_ica(dframe, rownum=0):
    '''Returns loaded raw and ica dataset from of dataframe'''
    row=dframe.iloc[rownum]
    raw = mne.io.read_raw_fif(row.raw_fname, preload=True)
    ica = mne.preprocessing.read_ica(row.ica_filename)
    return raw, ica
    
def plot_ecg_metrics():
    mous =  pd.read_csv('Spectra_MOUS.tsv', delimiter='\t')
    hcp = pd.read_csv('Spectra_HCP.tsv', delimiter='\t')
    cam = pd.read_csv('Spectra_CAMCAN.tsv', delimiter='\t')
    
    
    
    pylab.plot(cam[ca])
    
    

