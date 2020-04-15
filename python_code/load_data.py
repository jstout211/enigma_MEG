#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:08:12 2020

@author: stoutjd
"""
import os
import mne

mne.viz.set_3d_backend('pyvista')

#@save_numpy_output
def load_test_data():
    from hv_proc import test_config
    filename=test_config.rest['meg']
    raw=load_data(filename)
    return raw

def save_numpy_output(func):
    def wrapper(*args,**kwargs):
        #print("Something is happening before the function is called.")
        output=func(*args,**kwargs)
        output.save(func.__name__+'.npy')
        print("Saved data to {}".format(func.__name__+'.npy'))
    return wrapper

################

def check_datatype(filename):
    '''Check datatype based on the vendor naming convention'''
    if os.path.splitext(filename)[-1] == '.ds':
        return 'ctf'
    elif os.path.splitext(filename)[-1] == '.fif':
        return 'elekta'
    elif os.path.splitext(filename)[-1] == '.4d':
        return '4d'
    elif os.path.splitext(filename)[-1] == '.sqd':
        return 'kit'
    else:
        raise ValueError('Could not detect datatype')
        
def return_dataloader(datatype):
    '''Return the dataset loader for this dataset'''
    if datatype == 'ctf':
        return mne.io.read_raw_ctf
    if datatype == 'elekta':
        return mne.io.read_raw_fif
    if datatype == '4d':
        return mne.io.read_raw_bti
    if datatype == 'kit':
        return mne.io.read_raw_kit

def load_data(filename):
    datatype = check_datatype(filename)
    dataloader = return_dataloader(datatype)
    raw = dataloader(filename, preload=True)
    return raw

def calculate_inverse(epochs, outfolder=None):
    cov = mne.compute_covariance(epochs)
    cov.save(os.path.join(outfolder, 'rest-cov.fif'))
    
    
def visualize_coreg(raw, info, trans):
    mne.viz.set_3d_backend('pyvista')
    fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=info.subjid,
                                  subjects_dir=info.subjects_dir, surfaces='head',
                                  show_axes=True, meg='sensors',
                                  coord_frame='meg')
    mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))  
    # fig.plotter.show(screenshot='test.png')
    return fig  
    
    # mne.viz.set_3d_backend('pyvista')
    # fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=subjid,
    #                              subjects_dir=subjects_dir, surfaces='head',
    #                              show_axes=True, meg='sensors',
    #                              coord_frame='meg')
    # mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))    
    
def test_visualize_coreg():
    from hv_proc import test_config
    raw_fname = test_config.rest['meg']
    raw = mne.io.read_raw_ctf(raw_fname)
    import pickle
    from enigma.python_code.process_anatomical import anat_info
    enigma_dir=os.environ['ENIGMA_REST_DIR']
    with open(os.path.join(enigma_dir,'APBWVFAR_fs_ortho','info.pkl'),'rb') as e:
        info=pickle.load(e)
    trans=mne.transforms.Transform('mri', 'head')
    tmp = visualize_coreg(raw, info, trans=trans)
    

    
    # trans_fname=subjpd.loc[data_index, 'Anat_Coreg']
    # src_fname=subjpd.loc[data_index, 'Anat_SRC']
    # bem_sol_fname=subjpd.loc[data_index, 'Anat_BEM_SOL']
    # er_fname=subjpd.loc['emptyroom_1', 'PROC_INIT']
    # fwd_fname=subjpd.loc['RestEO_'+str(RUN),'Anat_FWD_SOL']
    # freq_bands=PROJECT.params['data'][DATATYPE]['freq_bands']
    # SUBJECTS_DIR=os.path.join(PROJECT.project_dir,SUBJID)
    # fs_subj='Anatomy'
    
    # fwd = mne.read_forward_solution(fwd_fname)
    # raw_fname=subjpd.ix[data_index, COLUMN]
    # raw = mne.io.Raw(raw_fname, preload=True)
    # picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, exclude='bads')
    # raw.pick_types(meg=True, eeg=False, stim=False, exclude='bads')

    # ##Create the inverse operator 
    # inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, 
    #                                                           loose=0.2)
    
    
def label_psd(epoch_vector, fs=None):
    '''Calculate the source level power spectral density from the label epochs'''
    from scipy.signal import welch
    freq_bins, epoch_spectra =  welch(epoch_vector, fs=fs, window='hanning') #, nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    return freq_bins, np.median(epoch_spectra, axis=0) #welch(epoch_vector, fs=fs, window='hanning') #, nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)    

# def test_label_psd():
#     freq_bins, spectral_power = label_psd(label_stack[:,1,:], 300)

def frequency_band_mean(label_by_freq=None, freq_band_list=None):
    '''Calculate the mean within the frequency bands'''
    for freqs in freq_band_list:
        label_by_freq()

def get_freq_idx(bands, freq_bins):
    ''' Get the frequency indexes'''
    output=[]
    for band in bands:
        tmp = np.argwhere((band[0] < freq_bins) & (freq_bins < band[1]))   ### <<<<<<<<<<<<< Should this be =<...
        output.append(tmp)
    return output



def main(filename):
    raw=load_data(filename)
    raw.apply_gradient_compensation(3)
    
    raw.resample(300)
    raw.filter(0.3, None)
    raw.notch_filter([60,120])
    
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)
    
    
    #Clear memory
    #del raw
    
    #Drop bad epochs and channels
    ##
    
    #Calculate covariance
    cov = mne.compute_covariance(epochs)
    
    # Load transformation matrix  << HACK for orthohulled data
    # trans=mne.transforms.Transform('ctf_head', 'ctf_meg')
    trans=mne.transforms.Transform('mri', 'head')   
    offset = np.array([-.5, -.5, -52.5])*0.01   ##### <<<<<<<<<<<<<<<  Fix - using single subject offset
    trans['trans'][0:3,-1] = offset
    
    
    HOME=os.environ['HOME']
    subjid='APBWVFAR_fs_ortho'
#    subjid = 'NLNHDGDO_fs_ortho'
    src = mne.read_source_spaces(os.path.join(HOME, 'hv_proc/enigma_outputs/'+subjid+'/source_space-src.fif'))
    bem = mne.read_bem_solution(os.path.join(HOME, 'hv_proc/enigma_outputs/'+subjid+'/bem_sol-sol.fif'))



    fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
    
    
    
    
    #Filter data
        
    
    
    
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, 
                                                           loose=0.2)
    #Calculate Inverse solution
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = 'MNE' #"dSPM"  # use dSPM method (could also be MNE or sLORETA)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                pick_ori="normal", return_generator=True)
    
    data_info = epochs.info
    

    SUBJECTS_DIR=os.environ['SUBJECTS_DIR']
    labels_lh=mne.read_labels_from_annot(subjid, parc='aparc',
                                        subjects_dir=SUBJECTS_DIR, hemi='lh') 
    labels_rh=mne.read_labels_from_annot(subjid, parc='aparc',
                                        subjects_dir=SUBJECTS_DIR, hemi='rh') 
    labels=labels_lh + labels_rh 
    label_ts=mne.extract_label_time_course(stcs, labels, src, mode='pca_flip') 
    
    label_stack = np.stack(label_ts)
    
    
    label_power = np.zeros([len(labels), len(freq_bins)])  #<< This will fail because freq_bins defined after
    
    #label_psd = np.zeros([100)
    for label_idx in range(len(labels)):
        # tmp = label_psd(label_stack[:,label_idx, :], raw.fs)
        
        _, label_power[label_idx,:] = label_psd(label_stack[:,label_idx, :], data_info['sfreq'])
    
    relative_power = label_power / label_power.sum(axis=1, keepdims=True)

    
    bands = [[1,3], [3,5], [7,12], [13,35], [35,55]]
    band_idxs = get_freq_idx(bands, freq_bins)

    band_means = np.zeros([len(labels), len(bands)]) 
    for mean_idx, band_idx in enumerate(band_idxs):
        band_means[mean_idx,:] = relative_power[:, band_idx].sum(axis=1)
    band_means = relative_power[:,]
        
    
        
    visualize_coreg(raw, info, trans=trans)
    
    

    


 # Check data type and load data
 # Downsample to 200Hz
 # Split to 1 second epochs
 # Reject sensor level data at a specific threshold
 # Calculate broad band dSPM inverse solution
 # Filter the data into bands (1-3, 3-6, 8-12, 13-35, 35-55)
 # Project the data to parcels and create parcel time series
 # Calculate relative power in each band and parcel    
    
   
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjects_dir', help='''Freesurfer subjects_dir can be 
                        assigned at the commandline if not already exported.''')
    parser.add_argument('-subjid', help='''Define subjects id (folder name)
                        in the SUBJECTS_DIR''')
    parser.add_argument('-meg_file', help='''Location of meg rest dataset''')
    
    args=parser.parse_args()
    
    #Load the anatomical information
    import pickle
    from enigma.python_code.process_anatomical import anat_info
    enigma_dir=os.environ['ENIGMA_REST_DIR']
    with open(os.path.join(enigma_dir,args.subjid,'info.pkl'),'rb') as e:
        info=pickle.load(e)
        
    raw=load_data(args.meg_file)
    trans=mne.transforms.Transform('mri', 'head')
    
    offset_cmd = 'mri_info --cras {}'.format(os.path.join(args.subjects_dir, subjid, 'mri', 'orig','001.mgz')))

    offset = np.array([-.5, -.5, -52.5])*0.01
    trans['trans'][0:3,-1] = offset
    
    visualize_coreg(raw, info, trans=trans)
    
        





