#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:08:12 2020

TODO:
    Add cleaning algorithm
    Check inverse solution
    Verify sum versus mean on bandwidth
    Check number of bins during welch calculation
    Type of covariance
    


@author: stoutjd
"""
import os
import os.path as op
import mne
from mne import Report
import hcp
import numpy as np
import pandas as pd
from enigmeg.spectral_peak_analysis import calc_spec_peak
from enigmeg.mod_label_extract import mod_source_estimate


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
    
def label_psd(epoch_vector, fs=None):
    '''Calculate the source level power spectral density from the label epochs'''
    # from scipy.signal import welch
    # freq_bins, epoch_spectra =  welch(epoch_vector, fs=fs, window='hanning') 
    
    from mne.time_frequency.multitaper import psd_array_multitaper
    epoch_spectra, freq_bins = psd_array_multitaper(epoch_vector, 
                                                    fs, 
                                                    fmin=1, fmax=45,
                                                    bandwidth=2, 
                                                    n_jobs=1, 
                                                    adaptive=True, 
                                                    low_bias=True) 
    
    return freq_bins, np.median(epoch_spectra, axis=0) 

def frequency_band_mean(label_by_freq=None, freq_band_list=None):
    '''Calculate the mean within the frequency bands'''
    for freqs in freq_band_list:
        label_by_freq()

def get_freq_idx(bands, freq_bins):
    ''' Get the frequency indexes'''
    output=[]
    for band in bands:
        tmp = np.nonzero((band[0] < freq_bins) & (freq_bins < band[1]))[0]   ### <<<<<<<<<<<<< Should this be =<...
        output.append(tmp)
    return output

def parse_proc_inputs(proc_file):
    # Load csv processing tab separated file
    proc_dframe = pd.read_csv(proc_file, sep='\t')    
    
    # Reject subjects with ignore flags
    keep_idx = proc_dframe.ignore.isna()   #May want to make a list of possible ignores
    proc_dframe = proc_dframe[keep_idx]
    
    for idx, dseries in proc_dframe.iterrows():
        print(dseries)
        
        dseries['output_dir']=op.expanduser(dseries['output_dir'])
        
        from types import SimpleNamespace
        info = SimpleNamespace()
        info.SUBJECTS_DIR = dseries['fs_subjects_dir']
        
        info.outfolder = op.join(dseries['output_dir'], dseries['subject'])
        info.bem_sol_filename = op.join(info.outfolder, 'bem_sol-sol.fif') 
        info.src_filename = op.join(info.outfolder, 'source_space-src.fif')
        
        os.environ['SUBJECTS_DIR']=dseries['fs_subjects_dir']
        
        #Determine if meg_file_path is a full path or relative path
        if not op.isabs(dseries['meg_file_path']):
            if op.isabs(dseries['meg_top_dir']):
                dseries['meg_file_path'] = op.join(dseries['meg_top_dir'], 
                                                   dseries['meg_file_path'])
            else:
                raise ValueError('This is not a valid path')
        
        #Perform the same check on the emptyroom data
        if not op.isabs(dseries['eroom_file_path']):
            if op.isabs(dseries['meg_top_dir']):
                dseries['eroom_file_path'] = op.join(dseries['meg_top_dir'], 
                                                   dseries['eroom_file_path'])
            else:
                raise ValueError('This is not a valid path')        
            
        inputs = {'filename' : dseries['meg_file_path'],
                  'subjid' : dseries['subject'],
                  'trans' : dseries['trans_file'],
                  'info' : info ,
                  'line_freq' : dseries['line_freq'],
                  'emptyroom_filename' : dseries['eroom_file_path']}
        main(**inputs)

def plot_QA_head_sensor_align(info, raw, trans):
    '''Plot and save the head and sensor alignment and save to the output folder'''
    from mayavi import mlab
    mlab.options.offscreen = True
    mne.viz.set_3d_backend('mayavi')
    
    outfolder = info.outfolder
    subjid = info.subjid
    subjects_dir = info.subjects_dir
    
    fig = mne.viz.plot_alignment(raw.info, trans, subject=subjid, dig=False,
                     coord_frame='meg', subjects_dir=subjects_dir)
    mne.viz.set_3d_view(figure=fig, azimuth=0, elevation=0)
    fig.scene.save_png(op.join(outfolder, 'lhead_posQA.png'))
    
    mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90)
    fig.scene.save_png(op.join(outfolder, 'rhead_posQA.png'))
    
    mne.viz.set_3d_view(figure=fig, azimuth=0, elevation=90)
    fig.scene.save_png(op.join(outfolder, 'front_posQA.png'))


def make_report(subject, subjects_dir, meg_filename, output_dir):
    #Create report from output
    report = Report(image_format='png', subjects_dir=subjects_dir,
                   subject=subject,  
                    raw_psd=False)  # use False for speed here
    #info_fname=meg_filename,  
    
    report.parse_folder(output_dir, on_error='ignore', mri_decim=10)
    report_filename = op.join(output_dir, 'QA_report.html')
    report.save(report_filename)
        
def test_beamformer():
   
    #Load filenames from test datasets
    from enigmeg.test_data.get_test_data import datasets
    test_dat = datasets().ctf

    meg_filename = test_dat['meg_rest'] 
    subjid = test_dat['subject']
    subjects_dir = test_dat['SUBJECTS_DIR'] 
    trans_fname = test_dat['trans']
    src_fname = test_dat['src']
    bem = test_dat['bem']
    
    outfolder = './tmp'  #<<< Change this ############################
    
    raw = mne.io.read_raw_ctf(meg_filename, preload=True)
    trans = mne.read_trans(trans_fname)
    # info.subjid, info.subjects_dir = subjid, subjects_dir
    
    raw.apply_gradient_compensation(3)
    raw.resample(300)
    raw.filter(1.0, None)
    raw.notch_filter([60,120])
    eraw.notch_filter([60,120])
    
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)

    data_cov = mne.compute_covariance(epochs, method='empirical')  
    
    eroom_filename = test_dat['meg_eroom'] 
    eroom_raw = mne.io.read_raw_ctf(eroom_filename, preload=True)
    eroom_raw.resample(300)
    eroom_raw.notch_filter([60,120])
    eroom_raw.filter(1.0, None)
    
    eroom_epochs = mne.make_fixed_length_epochs(eroom_raw, duration=4.0)
    noise_cov = mne.compute_covariance(eroom_epochs)

    fwd = mne.make_forward_solution(epochs.info, trans, src_fname, 
                                    bem)
    
    from mne.beamformer import make_lcmv, apply_lcmv_epochs
    filters = make_lcmv(epochs.info, fwd, data_cov, reg=0.01,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=None)
    
    labels_lh=mne.read_labels_from_annot(subjid, parc='aparc',
                                        subjects_dir=subjects_dir, hemi='lh') 
    labels_rh=mne.read_labels_from_annot(subjid, parc='aparc',
                                        subjects_dir=subjects_dir, hemi='rh') 
    labels=labels_lh + labels_rh 
    
    # labels[1].center_of_mass()
    
    results_stcs = apply_lcmv_epochs(epochs, filters, return_generator=True)#, max_ori_out='max_power')
    
    #Monkey patch of mne.source_estimate to perform 15 component SVD
    label_ts = mod_source_estimate.extract_label_time_course(results_stcs, labels, 
                                                         fwd['src'],
                                       mode='pca15_multitaper')
    
    #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
    label_stack = np.stack(label_ts)
    # label_stack = np.mean(label_stack, axis=0)

#    freq_bins, _ = label_psd(label_stack[:,0, :], raw.info['sfreq'])
    freq_bins = np.linspace(1,45,177)    ######################################3######### FIX

    #Initialize 
    label_power = np.zeros([len(labels), len(freq_bins)])  
    alpha_peak = np.zeros(len(labels))
    
    #Create PSD for each label
    for label_idx in range(len(labels)):
        print(str(label_idx))
        current_psd = label_stack[:,label_idx, :].mean(axis=0) 
        label_power[label_idx,:] = current_psd
        
        spectral_image_path = os.path.join(outfolder, 'Spectra_'+
                                            labels[label_idx].name + '.png')

        try:
            tmp_fmodel = calc_spec_peak(freq_bins, current_psd, 
                            out_image_path=spectral_image_path)
            
            #FIX FOR MULTIPLE ALPHA PEAKS
            potential_alpha_idx = np.where((8.0 <= tmp_fmodel.peak_params[:,0] ) & \
                                    (tmp_fmodel.peak_params[:,0] <= 12.0 ) )[0]
            if len(potential_alpha_idx) != 1:
                alpha_peak[label_idx] = np.nan         #############FIX ###########################3 FIX     
            else:
                alpha_peak[label_idx] = tmp_fmodel.peak_params[potential_alpha_idx[0]][0]
        except:
            alpha_peak[label_idx] = np.nan  #Fix <<<<<<<<<<<<<<

def assess_bads(raw_fname, is_eroom=False):
    '''Code sampled from MNE python website
    https://mne.tools/dev/auto_tutorials/preprocessing/\
        plot_60_maxwell_filtering_sss.html'''
    from mne.preprocessing import find_bad_channels_maxwell
    raw = mne.io.read_raw_fif(raw_fname)
    if raw.times[-1] > 60.0:
        raw.crop(tmax=60)    
    raw.info['bads'] = []
    raw_check = raw.copy()
    if is_eroom==False:
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None,
            return_scores=True, verbose=True)
    else:
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None,
            return_scores=True, verbose=True, coord_frame="meg")        
    
    return {'noisy':auto_noisy_chs, 'flat':auto_flat_chs}            

def main(filename=None, subjid=None, trans=None, info=None, line_freq=None, 
         emptyroom_filename=None, subjects_dir=None):
    
    raw = hcp.read_raw(subjid, 'rest', hcp_path='/data/EnigmaMeg/HCP/HCP_MEG')
    raw.load_data()
    
    eraw = hcp.read_raw(subjid, 'noise_empty_room',hcp_path='/data/EnigmaMeg/HCP/HCP_MEG')
    eraw.load_data()
    
    hcp.preprocessing.apply_ref_correction(raw)
    hcp.preprocessing.apply_ref_correction(eraw)
    #Below may be useful for testing ICA components
    #ica_mat = hcp.read_ica(subjid, 'rest')
    #annotations_dict=hcp.read_annot(subjid, 'rest')
    #hcp.preprocessing.apply_ica_hcp(raw, ica_mat, annotations_dict['ica']['ecg_eog_ic'])
    
    ## Load and prefilter continuous data
    #raw=load_data(filename)
    #eraw=load_data(emptyroom_filename)
    
    if type(raw)==mne.io.ctf.ctf.RawCTF:
        raw.apply_gradient_compensation(3)
    
    ## Test SSS bad channel detection for non-Elekta data
    # !!!!!!!!!!!  Currently no finecal or crosstalk used  !!!!!!!!!!!!!!!
    # if filename[-3:]=='fif':
    #     raw_bads_dict = assess_bads(filename)
    #     eraw_bads_dict = assess_bads(emptyroom_filename, is_eroom=True)
        
    #     raw.info['bads']=raw_bads_dict['noisy'] + raw_bads_dict['flat']
    #     eraw.info['bads']=eraw_bads_dict['noisy'] + eraw_bads_dict['flat']
    
    resample_freq=300
    
    raw.resample(resample_freq)
    eraw.resample(resample_freq)
    
    raw.filter(0.5, 140)
    eraw.filter(0.5, 140)
    
    if line_freq==None:
        try:
            line_freq = raw.info['line_freq']  # this isn't present in all files
        except:
            raise(ValueError('Could not determine line_frequency'))
    notch_freqs = np.arange(line_freq, 
                            resample_freq/2, 
                            line_freq)
    raw.notch_filter(notch_freqs)
    
    
    ## Create Epochs and covariance 
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)
    epochs.apply_baseline(baseline=(0,None))
    cov = mne.compute_covariance(epochs)
    
    er_epochs=mne.make_fixed_length_epochs(eraw, duration=4.0, preload=True)
    er_epochs.apply_baseline(baseline=(0,None))
    er_cov = mne.compute_covariance(er_epochs)
    
    os.environ['SUBJECTS_DIR']=subjects_dir
    src = mne.read_source_spaces(info.src_filename)
    bem = mne.read_bem_solution(info.bem_sol_filename)
    fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
    
    data_info = epochs.info
    
    from mne.beamformer import make_lcmv, apply_lcmv_epochs
    filters = make_lcmv(epochs.info, fwd, cov, reg=0.01,
                        noise_cov=er_cov, pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=None)
    
    labels_lh=mne.read_labels_from_annot(subjid, parc='aparc_sub',
                                        subjects_dir=subjects_dir, hemi='lh') 
    labels_rh=mne.read_labels_from_annot(subjid, parc='aparc_sub',
                                        subjects_dir=subjects_dir, hemi='rh') 
    labels=labels_lh + labels_rh 
    
    results_stcs = apply_lcmv_epochs(epochs, filters, return_generator=True)#, max_ori_out='max_power')
    
    #Monkey patch of mne.source_estimate to perform 15 component SVD
    label_ts = mod_source_estimate.extract_label_time_course(results_stcs, 
                                                             labels, 
                                                             fwd['src'],
                                                             mode='pca15_multitaper')
    
    #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
    label_stack = np.stack(label_ts)

    #HACK HARDCODED FREQ BINS
    freq_bins = np.linspace(1,45,177)    ######################################3######### FIX

    #Initialize 
    label_power = np.zeros([len(labels), len(freq_bins)])  
    alpha_peak = np.zeros(len(labels))
    
    #Create PSD for each label
    for label_idx in range(len(labels)):
        print(str(label_idx))
        current_psd = label_stack[:,label_idx, :].mean(axis=0) 
        label_power[label_idx,:] = current_psd
        
        spectral_image_path = os.path.join(info.outfolder, 'Spectra_'+
                                            labels[label_idx].name + '.png')

        try:
            tmp_fmodel = calc_spec_peak(freq_bins, current_psd, 
                            out_image_path=spectral_image_path)
            
            #FIX FOR MULTIPLE ALPHA PEAKS
            potential_alpha_idx = np.where((8.0 <= tmp_fmodel.peak_params[:,0] ) & \
                                    (tmp_fmodel.peak_params[:,0] <= 12.0 ) )[0]
            if len(potential_alpha_idx) != 1:
                alpha_peak[label_idx] = np.nan         #############FIX ###########################3 FIX     
            else:
                alpha_peak[label_idx] = tmp_fmodel.peak_params[potential_alpha_idx[0]][0]
        except:
            alpha_peak[label_idx] = np.nan  #Fix <<<<<<<<<<<<<<    
        
    #Save the label spectrum to assemble the relative power
    freq_bin_names=[str(binval) for binval in freq_bins]
    label_spectra_dframe = pd.DataFrame(label_power, columns=[freq_bin_names])
    label_spectra_dframe.to_csv( os.path.join(info.outfolder, 'label_spectra.csv') , index=False)
    # with open(os.path.join(info.outfolder, 'label_spectra.npy'), 'wb') as f:
    #     np.save(f, label_power)
    
    relative_power = label_power / label_power.sum(axis=1, keepdims=True)

    #Define bands
    bands = [[1,3], [3,6], [8,12], [13,35], [35,55]]
    band_idxs = get_freq_idx(bands, freq_bins)

    #initialize output
    band_means = np.zeros([len(labels), len(bands)]) 
    #Loop over all bands, select the indexes assocaited with the band and average    
    for mean_band, band_idx in enumerate(band_idxs):
        band_means[:, mean_band] = relative_power[:, band_idx].mean(axis=1) 
    
    output_filename = os.path.join(info.outfolder, 'Band_rel_power.csv')
    

    bands_str = [str(i) for i in bands]
    label_names = [i.name for i in labels]
    
    output_dframe = pd.DataFrame(band_means, columns=bands_str, 
                                 index=label_names)
    output_dframe['AlphaPeak'] = alpha_peak
    output_dframe.to_csv(output_filename, sep='\t')    
        
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjects_dir', help='''Freesurfer subjects_dir can be 
                        assigned at the commandline if not already exported.''')
    parser.add_argument('-subjid', help='''Define subjects id (folder name)
                        in the SUBJECTS_DIR''')
    parser.add_argument('-meg_file', help='''Location of meg rest dataset''')
    parser.add_argument('-er_meg_file', help='''Emptyroom dataset assiated with meg
                        file''')
    parser.add_argument('-viz_coreg', help='''Open up a window to vizualize the 
                        coregistration between head surface and MEG sensors''',
                        action='store_true')
    parser.add_argument('-trans', help='''Transfile from mne python -trans.fif''')
    parser.add_argument('-line_f', help='''Line frequecy''', type=float)
    parser.add_argument('-proc_file', help='''Process file to batch submit 
                        subjects.  Use proc_template.csv as a template''')
    
    args=parser.parse_args()
    
    if args.proc_file:
        proc_file = args.proc_file
        parse_proc_inputs(proc_file)
        exit(0)
    
    if not args.subjects_dir:
        subjects_dir = os.environ['SUBJECTS_DIR']
    else:
        subjects_dir = args.subjects_dir
    subjid = args.subjid
    
    #Load the anatomical information
    import pickle
    from enigmeg.process_anatomical import anat_info
    enigma_dir=os.environ['ENIGMA_REST_DIR']
    with open(os.path.join(enigma_dir,subjid,'info.pkl'),'rb') as e:
        info=pickle.load(e)
        
    raw=load_data(args.meg_file)
    
    trans = mne.read_trans(args.trans)
        
    if args.viz_coreg:
        plot_QA_head_sensor_align(info, raw, trans)
        # visualize_coreg(raw, info, trans=trans)
        # _ = input('Enter anything to exit')
        exit(0)
    
    del raw
    main(args.meg_file, subjid=subjid, trans=trans, info=info, 
         line_freq=args.line_f, emptyroom_filename=args.er_meg_file,
         subjects_dir=subjects_dir)
    
    


    
