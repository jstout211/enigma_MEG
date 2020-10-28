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
    Verify power on welch calc << does it need 20log10()
    Relative Power - Over all regions or just the ROI specta 
    Fix alpha peak decision - if multiple peaks in alpha range?
    


@author: stoutjd
"""
import os
import mne, numpy as np
import pandas as pd
from enigma.python_code.spectral_peak_analysis import calc_spec_peak

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
    
# def test_visualize_coreg():
#     from hv_proc import test_config
#     raw_fname = test_config.rest['meg']
#     raw = mne.io.read_raw_ctf(raw_fname)
#     import pickle
#     from enigma.python_code.process_anatomical import anat_info
#     enigma_dir=os.environ['ENIGMA_REST_DIR']
#     with open(os.path.join(enigma_dir,'APBWVFAR_fs_ortho','info.pkl'),'rb') as e:
#         info=pickle.load(e)
#     trans=mne.transforms.Transform('mri', 'head')
#     tmp = visualize_coreg(raw, info, trans=trans)
    

def label_psd(epoch_vector, fs=None):
    '''Calculate the source level power spectral density from the label epochs'''
    from scipy.signal import welch
    freq_bins, epoch_spectra =  welch(epoch_vector, fs=fs, window='hanning') #, nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    return freq_bins, np.median(epoch_spectra, axis=0) #welch(epoch_vector, fs=fs, window='hanning') #, nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)    




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

def plot_QA_head_sensor_align(info, raw, trans):
    '''Plot and save the head and sensor alignment and save to the output folder'''
    import matplotlib, pylab
    import os.path as op
    matplotlib.use('Agg')  
    
    
    fig = mne.viz.plot_alignment(raw.info, trans, subject=info.subjid, dig=False,
                     coord_frame='meg', subjects_dir=info.subjects_dir)
    mne.viz.set_3d_view(figure=fig, azimuth=0, elevation=0)
    
    tmp_outfile_name = op.join(info.outfolder, 'lhead_posQA.png')
    pylab.imshow(fig.plotter.image).figure.savefig(tmp_outfile_name)
    
    # fig.plotter.image.tofile(op.join(info.outfolder, 'lhead_posQA.png'))
    mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90)
    tmp_outfile_name = op.join(info.outfolder, 'rhead_posQA.png')
    pylab.imshow(fig.plotter.image).figure.savefig(tmp_outfile_name)    
    
    # fig.plotter.image.tofile(op.join(info.outfolder, 'rhead_posQA.png'))
    mne.viz.set_3d_view(figure=fig, azimuth=0, elevation=90)
    tmp_outfile_name = op.join(info.outfolder, 'front_posQA.png')
    pylab.imshow(fig.plotter.image).figure.savefig(tmp_outfile_name)  
    
    # fig.plotter.image.tofile(op.join(info.outfolder, 'front_posQA.png'))
    # test = input('Press any key to close')
    fig.plotter.close()

def test_QA_plot():
    import bunch
    info = bunch.Bunch()
    meg_filename = '/home/stoutjd/data/MEG/20190115/AYCYELJY_rest_20190115_03.ds'
    subjid = 'AYCYELJY_fs'
    subjects_dir = '/home/stoutjd/data/ENIGMA'
    raw = mne.io.read_raw_ctf(meg_filename)
    trans = mne.read_trans('/home/stoutjd/data/ENIGMA/transfiles/AYCYELJY-trans.fif')
    info.subjid, info.subjects_dir = subjid, subjects_dir
    plot_QA_head_sensor_align(info, raw, trans ) #subjid, subjects_dir)
    

def test_main():
    HOME=os.environ['HOME']
    filename = os.path.join(HOME,'hv_proc/MEG/APBWVFAR_rest_20200122_03.ds')
    subjid = 'APBWVFAR_fs_ortho'
    subjects_dir = os.path.join(HOME, 'hv_proc', 'MRI')
    
    # Calc Transform
    raw=load_data(filename)
    trans=mne.transforms.Transform('mri', 'head')
    
    # Get the MRI offset from freesurfer call
    offset_cmd = 'mri_info --cras {}'.format(os.path.join(subjects_dir, subjid, 
                                                          'mri', 'orig','001.mgz'))
    
    from subprocess import check_output
    offset = check_output(offset_cmd.split(' ')).decode()[:-1]
    offset = offset.split(' ')
    offset = np.array([float(i) for i in offset])
    
    offset[2] *= -1
    offset *= .001  #Convert to mm
    trans['trans'][0:3,-1] = offset
    
    import pickle
    from enigma.python_code.process_anatomical import anat_info
    enigma_dir=os.environ['ENIGMA_REST_DIR']
    with open(os.path.join(enigma_dir,subjid,'info.pkl'),'rb') as e:
        info=pickle.load(e)
    
    
  


def main(filename=None, subjid=None, trans=None, info=None):
    raw=load_data(filename)
    raw.apply_gradient_compensation(3)
    
    #plot_QA_head_sensor_align(info, raw)
    
    raw.resample(300)
    raw.filter(0.3, None)
    raw.notch_filter([60,120])
    
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)
    
    ####  Reduced for DEMO  ##############
    #epochs=epochs[0:10]
    
    #Drop bad epochs and channels
    ##
    
    #Calculate covariance
    cov = mne.compute_covariance(epochs)
    
    HOME=os.environ['HOME']
    src = mne.read_source_spaces(info.src_filename)
    #src = mne.read_source_spaces(os.path.join(HOME, 'hv_proc/enigma_outputs/'+subjid+'/source_space-src.fif'))
    # bem = mne.read_bem_solution(os.path.join(HOME, 'hv_proc/enigma_outputs/'+subjid+'/bem_sol-sol.fif'))
    bem = mne.read_bem_solution(info.bem_sol_filename)


    fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, 
                                                           loose=0.2)
    #Calculate Inverse solution
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = 'dSPM'  
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                pick_ori="normal", return_generator=True)
    
    data_info = epochs.info
    

    SUBJECTS_DIR=os.environ['SUBJECTS_DIR']
    labels_lh=mne.read_labels_from_annot(subjid, parc='aparc',
                                        subjects_dir=SUBJECTS_DIR, hemi='lh') 
    labels_rh=mne.read_labels_from_annot(subjid, parc='aparc',
                                        subjects_dir=SUBJECTS_DIR, hemi='rh') 
    labels=labels_lh + labels_rh 
    
    #labels = labels[0:10]  ######## <<< HACK for DEMO  3####################################
    
    label_ts=mne.extract_label_time_course(stcs, labels, src, mode='pca_flip') 
    
    #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
    label_stack = np.stack(label_ts)
    
    
    freq_bins, _ = label_psd(label_stack[:,0, :], data_info['sfreq'])
    
    #Initialize 
    label_power = np.zeros([len(labels), len(freq_bins)])  
    alpha_peak = np.zeros(len(labels))
    
    #Create PSD for each label
    for label_idx in range(len(labels)):
        _, current_psd = label_psd(label_stack[:,label_idx, :], 
                                                data_info['sfreq'])
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


    
    #label_power *= np.sqrt(freq_bins)  ###################### SCALE BY Frequency  <<<<<<<<<<<<<<<<< CHECK
    
    
    relative_power = label_power / label_power.sum(axis=1, keepdims=True)

    #Define bands
    bands = [[1,3], [3,6], [8,12], [13,35], [35,55]]
    band_idxs = get_freq_idx(bands, freq_bins)

    ############  MEAN or sum????????????

    #initialize output
    band_means = np.zeros([len(labels), len(bands)]) 
    #Loop over all bands, select the indexes assocaited with the band and average    
    for mean_band, band_idx in enumerate(band_idxs):
        band_means[:, mean_band] = relative_power[:, band_idx].mean(axis=1)   ##########  <<< mEAN or SUM
    
    output_filename = os.path.join(info.outfolder, 'Band_rel_power.csv')
    

    bands_str = [str(i) for i in bands]
    label_names = [i.name for i in labels]
    
    output_dframe = pd.DataFrame(band_means, columns=bands_str, 
                                 index=label_names)
    output_dframe['AlphaPeak'] = alpha_peak
    output_dframe.to_csv(output_filename, sep='\t')    
        


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
    parser.add_argument('-viz_coreg', help='''Open up a window to vizualize the 
                        coregistration between head surface and MEG sensors''',
                        action='store_true')
    parser.add_argument('-trans', help='''Transfile from mne python -trans.fif''')
    
    args=parser.parse_args()
    if not args.subjects_dir:
        subjects_dir = os.environ['SUBJECTS_DIR']
    else:
        subjects_dir = args.subjects_dir
    subjid = args.subjid
    
    #Load the anatomical information
    import pickle
    from enigma.python_code.process_anatomical import anat_info
    enigma_dir=os.environ['ENIGMA_REST_DIR']
    with open(os.path.join(enigma_dir,subjid,'info.pkl'),'rb') as e:
        info=pickle.load(e)
        
    raw=load_data(args.meg_file)
    
    trans = mne.read_trans(args.trans)
    
    # trans=mne.transforms.Transform('mri', 'head')
    
    # # Get the MRI offset from freesurfer call
    # offset_cmd = 'mri_info --cras {}'.format(os.path.join(subjects_dir, subjid, 
    #                                                       'mri', 'orig','001.mgz'))
    
    # from subprocess import check_output
    # offset = check_output(offset_cmd.split(' ')).decode()[:-1]
    # offset = offset.split(' ')
    # offset = np.array([float(i) for i in offset])
    
    # # Convert to RAS ????????????????????????  << Verify 
    # offset[2] *= -1
    # offset *= .001  #Convert to mm
    # trans['trans'][0:3,-1] = offset
    
    if args.viz_coreg:
        plot_QA_head_sensor_align(info, raw, trans)
        # visualize_coreg(raw, info, trans=trans)
        _ = input('Enter anything to exit')
        exit(0)
    
    del raw
    main(args.meg_file, subjid=subjid, trans=trans, info=info)
    
    
        





