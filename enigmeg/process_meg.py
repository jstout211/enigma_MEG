#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:08:12 2020


@author: stoutjd
"""
import os
import os.path as op
import mne
from mne import Report
import numpy as np
import pandas as pd
from enigmeg.spectral_peak_analysis import calc_spec_peak
from enigmeg.mod_label_extract import mod_source_estimate
import logging
import munch 
from collections import OrderedDict
import mne_bids
from mne_bids import get_head_mri_trans
from mne.beamformer import make_lcmv, apply_lcmv_epochs

logger=logging.basicConfig()
n_jobs = 5  #extract this from the configuration file
os.environ['n_jobs'] = str(n_jobs)

from mne_bids import BIDSPath
import functools

class process():
    def __init__(
            self, 
            subject=None, 
            bids_root=None, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname='rest',
            emptyroom_tagname='emptyroom',
            session=1, 
            mains=60,
            run='1',
            t1_override=None,
            fs_ave_fids=False
            ):
        
# =============================================================================
#         #Initialize variables
# =============================================================================
        self.subject=subject.replace('sub-','')  #Strip header if present
        self.bids_root=bids_root
        if deriv_root is None:
            self.deriv_root = op.join(
                bids_root, 
                'derivatives'
                )
        else:
            self.deriv_root = deriv_root
            
        self.enigma_root = op.join(
            self.deriv_root,
            'ENIGMA_MEG'
            )
        
        if subjects_dir is None:
            self.subjects_dir = op.join(
                self.deriv_root,
                'freesurfer',
                'subjects'
                )
        self.proc_vars=munch.Munch()
        self.proc_vars['fmin'] = 1
        self.proc_vars['fmax'] = 45
        self.proc_vars['sfreq'] = 300
        self.proc_vars['mains'] = mains
        self.proc_vars['epoch_len']=4.0  #seconds
        
        self._t1_override = t1_override
        if fs_ave_fids==True:
            self._use_fsave_coreg=True
        else:
            self._use_fsave_coreg=False
        
        
        
        
        
            
# =============================================================================
#             Configure bids paths
# =============================================================================
        self.bids_path = BIDSPath(
            root=bids_root, 
            subject=subject, 
            session=session)
        
        self.deriv_path=self.bids_path.copy().update(
            root=self.enigma_root,
            check=False,
            extension='.fif',
            datatype='meg', 
            suffix='meg'
            )
        
        if not op.exists(self.deriv_path.directory): 
            self.deriv_path.directory.mkdir(parents=True)
        
        self.rest_derivpath = self.deriv_path.copy().update(
            task=rest_tagname, 
            run=run
            )
        
        self.eroom_derivpath = self.deriv_path.copy().update(
            task=emptyroom_tagname
            )
        
        self.meg_rest_raw = self.bids_path.copy().update(
            datatype='meg', 
            task=rest_tagname, 
            run=run
            )
        
        self.meg_er_raw = self.bids_path.copy().update(
            datatype='meg',
            task=emptyroom_tagname
            )
        
        self.anat_bidspath = self.bids_path.copy().update(root=self.subjects_dir,
                                                          session=None,
                                                          check=False)
        
        self.check_paths()
        
        self.fnames=self.initialize_fnames(rest_tagname, emptyroom_tagname)
    
    def initialize_fnames(self, rest_tagname, emptyroom_tagname):
        '''Use the bids paths to generate output names'''
        _tmp=munch.Munch()
        rest_deriv = self.rest_derivpath.copy().update(extension='.fif')
        eroom_deriv = self.eroom_derivpath.copy().update(extension='.fif')
        
        ## Setup bids paths for all 
        # Conversion to actual paths at end
        
        _tmp['raw_rest']=self.meg_rest_raw
        _tmp['raw_eroom']=self.meg_er_raw
        
        _tmp['rest_filt']=rest_deriv.copy().update(processing='filt')
        _tmp['eroom_filt']=eroom_deriv.copy().update(processing='filt')
        
        # MEGNET post
        #_tmp['rest_clean']=rest_deriv.copy().update(processing='clean')
        #_tmp['eroom_clean']=eroom_deriv.copy().update(processing='clean')
        
        _tmp['rest_epo']=rest_deriv.copy().update(suffix='epo')
        _tmp['eroom_epo']=eroom_deriv.copy().update(suffix='epo')
        
        #_tmp['rest_epo_clean']=_tmp['rest_epo'].copy().update(processing='clean')
        #_tmp['eroom_epo_clean']=_tmp['eroom_epo'].copy().update(processing='clean')
        
        _tmp['rest_cov']=rest_deriv.copy().update(suffix='cov')
        _tmp['eroom_cov']=eroom_deriv.copy().update(suffix='cov')
        
        _tmp['rest_fwd']=rest_deriv.copy().update(suffix='fwd') 
        _tmp['rest_trans']=rest_deriv.copy().update(suffix='trans')
        _tmp['bem'] = self.deriv_path.copy().update(suffix='bem', extension='.fif')
        _tmp['src'] = self.deriv_path.copy().update(suffix='src', extension='.fif')
        
        _tmp['lcmv'] = self.deriv_path.copy().update(suffix='lcmv', extension='.h5')
        
        # Cast all bids paths to paths and save as dictionary
        path_dict = {key:str(i.fpath) for key,i in _tmp.items()}
        
        # Additional non-bids path files
        path_dict['parc'] = op.join(self.subjects_dir, 'morph-maps', 
                               f'sub-{self.subject}-fsaverage-morph.fif') 
        return munch.Munch(path_dict)
        
        
# =============================================================================
#       Load data
# =============================================================================
    def load_data(self):
        if not hasattr(self, 'raw_rest'):
            self.raw_rest = load_data(self.meg_rest_raw.fpath) 
            self.raw_rest.pick_types(meg=True, eeg=False)
        if not hasattr(self, 'raw_eroom'):
            self.raw_eroom = load_data(self.meg_er_raw.fpath) 
            self.raw_eroom.pick_types(meg=True, eeg=False)
        
    
    def check_paths(self):
        '''Verify that the raw data is present and can be found'''
        try:
            self.meg_rest_raw.fpath  #Errors if not present
            self.vendor, self.extra_params = check_datatype(str(self.meg_rest_raw.fpath))
        except:
            logging.exception(f'Could not find rest dataset:\n')
            
        try:
            self.meg_er_raw.fpath
        except:
            logging.exception(f'Could not find emptyroom dataset:\n')
        
        subj_fsdir= op.join(f'{self.subjects_dir}', f'sub-{self.subject}')
        if not op.exists(subj_fsdir):
            errtxt = f'There is no freesurfer folder for {self.subject}:\
                              \n{subj_fsdir}'.replace('  ',' ')
            logging.exception(errtxt)
                         
                         

# =============================================================================
#       Vendor specific prep
# =============================================================================
    def vendor_prep(self):
        '''Different vendor types require special cleaning / initialization'''
        ## Elekta/MEGIN
        if self.vendor == 'fif':
            rest_bads_ = assess_bads(self.meg_rest_raw.fpath)
            er_bads_ = assess_bads(self.meg_er_raw.fpath,
                                   is_errom=True)
            
            self.raw_rest.info['bads'] = rest_bads_['noisy'] + rest_bads_['flat']
            self.raw_eroom.info['bads'] = er_bads_['noisy'] + er_bads_['flat']
        
        ## CTF
        if self.vendor == 'ctf':
            if self.raw_rest.compensation_grade != 3:
                logging.info(f'Applying 3rd order gradient to rest data')
                self.apply_gradient_compensation(3)
            if self.raw_eroom.compensation_grade != 3:
                logging.info(f'Applying 3rd order gradient to emptyroom data')
                self.apply_gradient_compensation(3)
        
        ## 4D
        
        ## KIT 
        
            
# =============================================================================
#       Preprocessing
# =============================================================================
    def _preproc(self,
                raw_inst=None,
                deriv_path=None):
        raw_inst.resample(self.proc_vars['sfreq'], n_jobs=n_jobs)
        raw_inst.notch_filter(self.proc_vars['mains'], n_jobs=n_jobs) 
        raw_inst.filter(self.proc_vars['fmin'], self.proc_vars['fmax'], n_jobs=n_jobs)
        raw_inst.save(deriv_path.copy().update(processing='filt'), overwrite=True)

    def do_preproc(self):
        '''Preprocess both datasets'''
        self._preproc(raw_inst=self.raw_rest, deriv_path=self.rest_derivpath)
        self._preproc(raw_inst=self.raw_eroom, deriv_path=self.eroom_derivpath)
        
    def _proc_epochs(self,
                     raw_inst=None,
                     deriv_path=None):
        '''Create and save epochs
        Create and save covariance'''
        epochs = mne.make_fixed_length_epochs(raw_inst, 
                                              duration=self.proc_vars['epoch_len'], 
                                              preload=True)
        epochs_fname = deriv_path.copy().update(suffix='epo')
        epochs.save(epochs_fname, overwrite=True)
        
        cov = mne.compute_covariance(epochs)
        cov_fname = deriv_path.copy().update(suffix='cov')
        cov.save(cov_fname, overwrite=True)
        
    def do_proc_epochs(self):
        self._proc_epochs(raw_inst=self.raw_rest,
                          deriv_path=self.rest_derivpath)
        self._proc_epochs(raw_inst=self.raw_eroom, 
                          deriv_path=self.eroom_derivpath)
        
    def proc_mri(self, t1_override=None,redo_all=False):
        if t1_override is not None:
            entities=mne_bids.get_entities_from_fname(t1_override)
            t1_bids_path = BIDSPath(**entities)
        else:
            t1_bids_path = self.bids_path.copy().update(datatype='anat', 
                                                    suffix='T1w')
                                                    
        # Configure filepaths
        os.environ['SUBJECTS_DIR']=self.subjects_dir
        deriv_path = self.rest_derivpath.copy()
        # Update path so src/bem are not saved as task/run specific
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif',
                                             task=None, run=None)
        bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif',
                                             task=None, run=None)
        # Specific to task = rest / run = run#
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')
                
        if fwd_fname.fpath.exists() and (redo_all is False):
            fwd = mne.read_forward_solution(fwd_fname)
            self.rest_fwd = fwd
        
        watershed_check_path=op.join(self.subjects_dir,
                                     f'sub-{self.subject}',
                                     'bem',
                                     'inner_skull.surf'
                                     )
        if (not os.path.exists(watershed_check_path)) or (redo_all is True):
            mne.bem.make_watershed_bem(f'sub-{self.subject}',
                                       self.subjects_dir,
                                       overwrite=True
                                       )
        if (not bem_fname.fpath.exists()) or (redo_all is True):
            bem = mne.make_bem_model(f'sub-{self.subject}', 
                                     subjects_dir=self.subjects_dir, 
                                     conductivity=[0.3])
            bem_sol = mne.make_bem_solution(bem)
            
            mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
        else:
            bem_sol = mne.read_bem_solution(bem_fname)
            
        if (not src_fname.fpath.exists()) or (redo_all is True):
            src = mne.setup_source_space(f'sub-{self.subject}',
                                         spacing='oct6', add_dist='patch',
                                 subjects_dir=self.subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
        
        if (not trans_fname.fpath.exists()) or (redo_all is True):
            if self._use_fsave_coreg==True:
                trans = self.do_auto_coreg()
            else:
                trans = get_head_mri_trans(self.meg_rest_raw,
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject) 
            mne.write_trans(trans_fname.fpath, trans, overwrite=True)
        else:
            trans = mne.read_trans(trans_fname.fpath)
        fwd = mne.make_forward_solution(self.raw_rest.info, trans, src, bem_sol, eeg=False, 
                                        n_jobs=n_jobs)
        self.rest_fwd=fwd
        mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)
        
    def do_auto_coreg(self):
        '''Localize coarse fiducials based on fsaverage coregistration
        and fine tuned with iterative headshape fit.'''
        coreg = mne.coreg.Coregistration(self.raw_rest.info, 
                                         f'sub-{self.subject}', 
                                         subjects_dir=self.subjects_dir, 
                                         fiducials='estimated')
        coreg.fit_fiducials(verbose=True)
        coreg.omit_head_shape_points(distance=5. / 1000)  # distance is in meters
        coreg.fit_icp(n_iterations=6, nasion_weight=.5, hsp_weight= 5, verbose=True)
        return coreg.trans
            
    
    def do_make_aparc_sub(self):
        write_aparc_sub(subjid=f'sub-{self.subject}', 
                        subjects_dir=self.subjects_dir)
        
    def do_beamformer(self):
        dat_cov = mne.read_cov(self.fnames.rest_cov)
        forward = mne.read_forward_solution(self.fnames.rest_fwd)
        epochs = mne.read_epochs(self.fnames.rest_epo)
        fname_lcmv = self.fnames.lcmv #Pre-assign output name
        
        # If emptyroom present - use in beamformer
        if hasattr(self, 'meg_er_raw'):
            noise_cov = mne.read_cov(self.fnames.eroom_cov)
            noise_rank = mne.compute_rank(self.raw_eroom)
            filters = make_lcmv(epochs.info, forward, dat_cov, reg=0.05, 
                    noise_cov=noise_cov,  pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=noise_rank)
        else:
            #Build beamformer without emptyroom noise
            epo_rank = mne.compute_rank(epochs)
            filters = make_lcmv(epochs.info, forward, dat_cov, reg=0.05, 
                            pick_ori='max-power',
                            weight_norm='unit-noise-gain', rank=epo_rank)
        
        filters.save(fname_lcmv, overwrite=True)
        stcs = apply_lcmv_epochs(epochs, filters, return_generator=True) 
        self.stcs = stcs
        
    def do_label_psds(self):
        labels_lh=mne.read_labels_from_annot(f'sub-{self.subject}',
                                             parc='aparc_sub',
                                            subjects_dir=self.subjects_dir,
                                            hemi='lh') 
        labels_rh=mne.read_labels_from_annot(f'sub-{self.subject}',
                                             parc='aparc_sub',
                                             subjects_dir=self.subjects_dir,
                                             hemi='rh') 
        labels=labels_lh + labels_rh 
        self.labels = labels
        
        #Monkey patch of mne.source_estimate to perform 15 component SVD
        label_ts = mod_source_estimate.extract_label_time_course(self.stcs, 
                                                                 labels, 
                                                                 self.rest_fwd['src'],
                                                                 mode='pca15_multitaper')
        
        #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
        self.label_ts = np.stack(label_ts)
        
    def do_spectral_paramerization(self):
        '''
        Passes spectra to fooof alg for peak and 1/f analysis

        Returns
        -------
        None.

        '''
        #HACK HARDCODED FREQ BINS
        freq_bins = np.linspace(1,45,177)    ######################################3######### FIX
    
        #Initialize 
        labels = self.labels
        label_power = np.zeros([len(labels), len(freq_bins)])  
        alpha_peak = np.zeros(len(labels))
        
        outfolder = self.deriv_path.directory / 'foof_results'
        self.results_dir = outfolder
        if not os.path.exists(outfolder): os.mkdir(outfolder)
        
        #Create PSD for each label
        label_stack = self.label_ts
        for label_idx in range(len(self.labels)):
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
            
        #Save the label spectrum to assemble the relative power
        freq_bin_names=[str(binval) for binval in freq_bins]
        label_spectra_dframe = pd.DataFrame(label_power, columns=[freq_bin_names])
        label_spectra_dframe.to_csv( os.path.join(outfolder, 'label_spectra.csv') , index=False)
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
        
        output_filename = os.path.join(outfolder, 'Band_rel_power.csv')
        
    
        bands_str = [str(i) for i in bands]
        label_names = [i.name for i in labels]
        
        output_dframe = pd.DataFrame(band_means, columns=bands_str, 
                                     index=label_names)
        output_dframe['AlphaPeak'] = alpha_peak
        output_dframe.to_csv(output_filename, sep='\t')    



        
        
    def list_outputs(self):
        exists = [i for i in self.fnames if op.exists(self.fnames[i])]
        missing = [i for i in self.fnames if not op.exists(self.fnames[i])]
        for i in exists:
            print(f'Present: {self.fnames[i]}')
        print('\n\n')
        for i in missing:
            print(f'Missing: {self.fnames[i]}')
        
    def do_proc_allsteps(self):
        self.load_data()
        self.do_preproc()
        self.do_proc_epochs()
        self.proc_mri(t1_override=self._t1_override)
        self.do_beamformer()
        self.do_make_aparc_sub()
        self.do_label_psds()
        self.do_spectral_paramerization()
        
    
    def check_alignment(self):
        self.trans = mne.read_trans(self.fnames['rest_trans'])
        self.load_data()
        mne.viz.plot_alignment(self.raw_rest.info,
                                    trans=self.trans,
                                    subject='sub-'+self.bids_path.subject,
                                    subjects_dir=self.subjects_dir,
                                    dig=True)
        
        
      

def check_datatype(filename):
    '''Check datatype based on the vendor naming convention'''
    if os.path.splitext(filename)[-1] == '.ds':
        return 'ctf', dict(system_clock='ignore')
    elif os.path.splitext(filename)[-1] == '.fif':
        return 'elekta', dict(allow_maxshield=True)
    elif os.path.splitext(filename)[-1] == '.4d':
        return '4d', None
    elif os.path.splitext(filename)[-1] == '.sqd':
        return 'kit', None
    elif os.path.splitext(filename)[-1] == 'con':
        return 'kit', None
    else:
        raise ValueError('Could not detect datatype')
        
def return_dataloader(datatype):
    '''Return the dataset loader for this dataset'''
    if datatype == 'ctf':
        return functools.partial(mne.io.read_raw_ctf, system_clock='ignore')
    if datatype == 'elekta':
        return mne.io.read_raw_fif
    if datatype == '4d':
        return mne.io.read_raw_bti
    if datatype == 'kit':
        return mne.io.read_raw_kit

def load_data(filename):
    datatype, _ = check_datatype(filename)
    dataloader = return_dataloader(datatype)
    raw = dataloader(filename, preload=True)
    return raw

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

def write_aparc_sub(subjid=None, subjects_dir=None):
    '''Check for fsaverage and aparc_sub and download
    Morph fsaverage aparc_sub labels to single subject data
    
    https://mne.tools/stable/auto_examples/visualization/plot_parcellation.html
    '''
    if not op.exists(op.join(subjects_dir, 'fs_average')):
        mne.datasets.fetch_fsaverage(verbose='ERROR') #True requires TQDM
    if not op.exists(op.join(subjects_dir, 'fs_average', 'label', 
                             'lh.aparc_sub.annot')):
        mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose='ERROR')
    
    sub_labels=mne.read_labels_from_annot('fsaverage',parc='aparc_sub', 
                                   subjects_dir=subjects_dir)        
    subject_labels=mne.morph_labels(sub_labels, subject_to=subjid, 
                                 subjects_dir=subjects_dir)
    mne.write_labels_to_annot(subject_labels, subject=subjid, 
                              parc='aparc_sub', subjects_dir=subjects_dir, 
                              overwrite=True)





def load_test_data(**kwargs):
    proc = process(subject='ON02747',
                        bids_root=op.expanduser('~/ds004215'),
                        session='01',
                        emptyroom_tagname='noise', 
                        mains=60,
                        t1_override='~/ds004215/sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
                        **kwargs)
    # proc.load_data()
    return proc

# print(__name__)
# # For Testing run this cell
# if __name__!='__main__':
# proc = load_test_data(run='01')



# !!! Fix hardcoded variables   
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


#%%
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
        


# def main(filename=None, subjid=None, trans=None, info=None, line_freq=None, 
#          emptyroom_filename=None, subjects_dir=None):
    
#     ## Load and prefilter continuous data
#     raw=load_data(filename)
#     eraw=load_data(emptyroom_filename)
    
#     if type(raw)==mne.io.ctf.ctf.RawCTF:
#         raw.apply_gradient_compensation(3)
    
#     ## Test SSS bad channel detection for non-Elekta data
#     # !!!!!!!!!!!  Currently no finecal or crosstalk used  !!!!!!!!!!!!!!!
#     if filename[-3:]=='fif':
#         raw_bads_dict = assess_bads(filename)
#         eraw_bads_dict = assess_bads(emptyroom_filename, is_eroom=True)
        
#         raw.info['bads']=raw_bads_dict['noisy'] + raw_bads_dict['flat']
#         eraw.info['bads']=eraw_bads_dict['noisy'] + eraw_bads_dict['flat']
    
#     resample_freq=300
    
#     raw.resample(resample_freq)
#     eraw.resample(resample_freq)
    
#     raw.filter(0.5, 140)
#     eraw.filter(0.5, 140)
    
#     if line_freq==None:
#         try:
#             line_freq = raw.info['line_freq']  # this isn't present in all files
#         except:
#             raise(ValueError('Could not determine line_frequency'))
#     notch_freqs = np.arange(line_freq, 
#                             resample_freq/2, 
#                             line_freq)
#     raw.notch_filter(notch_freqs)
    
    
#     ## Create Epochs and covariance 
#     epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)
#     epochs.apply_baseline(baseline=(0,None))
#     cov = mne.compute_covariance(epochs)
    
#     er_epochs=mne.make_fixed_length_epochs(eraw, duration=4.0, preload=True)
#     er_epochs.apply_baseline(baseline=(0,None))
#     er_cov = mne.compute_covariance(er_epochs)
    
#     os.environ['SUBJECTS_DIR']=subjects_dir
#     src = mne.read_source_spaces(info.src_filename)
#     bem = mne.read_bem_solution(info.bem_sol_filename)
#     fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
    
#     data_info = epochs.info
    
#     from mne.beamformer import make_lcmv, apply_lcmv_epochs
#     filters = make_lcmv(epochs.info, fwd, cov, reg=0.01,
#                         noise_cov=er_cov, pick_ori='max-power',
#                         weight_norm='unit-noise-gain', rank=None)
    
#     labels_lh=mne.read_labels_from_annot(subjid, parc='aparc_sub',
#                                         subjects_dir=subjects_dir, hemi='lh') 
#     labels_rh=mne.read_labels_from_annot(subjid, parc='aparc_sub',
#                                         subjects_dir=subjects_dir, hemi='rh') 
#     labels=labels_lh + labels_rh 
    
#     results_stcs = apply_lcmv_epochs(epochs, filters, return_generator=True)#, max_ori_out='max_power')
    
#     #Monkey patch of mne.source_estimate to perform 15 component SVD
#     label_ts = mod_source_estimate.extract_label_time_course(results_stcs, 
#                                                              labels, 
#                                                              fwd['src'],
#                                                              mode='pca15_multitaper')
    
#     #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
#     label_stack = np.stack(label_ts)

#     #HACK HARDCODED FREQ BINS
#     freq_bins = np.linspace(1,45,177)    ######################################3######### FIX

#     #Initialize 
#     label_power = np.zeros([len(labels), len(freq_bins)])  
#     alpha_peak = np.zeros(len(labels))
    
#     #Create PSD for each label
#     for label_idx in range(len(labels)):
#         print(str(label_idx))
#         current_psd = label_stack[:,label_idx, :].mean(axis=0) 
#         label_power[label_idx,:] = current_psd
        
#         spectral_image_path = os.path.join(info.outfolder, 'Spectra_'+
#                                             labels[label_idx].name + '.png')

#         try:
#             tmp_fmodel = calc_spec_peak(freq_bins, current_psd, 
#                             out_image_path=spectral_image_path)
            
#             #FIX FOR MULTIPLE ALPHA PEAKS
#             potential_alpha_idx = np.where((8.0 <= tmp_fmodel.peak_params[:,0] ) & \
#                                     (tmp_fmodel.peak_params[:,0] <= 12.0 ) )[0]
#             if len(potential_alpha_idx) != 1:
#                 alpha_peak[label_idx] = np.nan         #############FIX ###########################3 FIX     
#             else:
#                 alpha_peak[label_idx] = tmp_fmodel.peak_params[potential_alpha_idx[0]][0]
#         except:
#             alpha_peak[label_idx] = np.nan  #Fix <<<<<<<<<<<<<<    
        
#     #Save the label spectrum to assemble the relative power
#     freq_bin_names=[str(binval) for binval in freq_bins]
#     label_spectra_dframe = pd.DataFrame(label_power, columns=[freq_bin_names])
#     label_spectra_dframe.to_csv( os.path.join(info.outfolder, 'label_spectra.csv') , index=False)
#     # with open(os.path.join(info.outfolder, 'label_spectra.npy'), 'wb') as f:
#     #     np.save(f, label_power)
    
#     relative_power = label_power / label_power.sum(axis=1, keepdims=True)

#     #Define bands
#     bands = [[1,3], [3,6], [8,12], [13,35], [35,55]]
#     band_idxs = get_freq_idx(bands, freq_bins)

#     #initialize output
#     band_means = np.zeros([len(labels), len(bands)]) 
#     #Loop over all bands, select the indexes assocaited with the band and average    
#     for mean_band, band_idx in enumerate(band_idxs):
#         band_means[:, mean_band] = relative_power[:, band_idx].mean(axis=1) 
    
#     output_filename = os.path.join(info.outfolder, 'Band_rel_power.csv')
    

#     bands_str = [str(i) for i in bands]
#     label_names = [i.name for i in labels]
    
#     output_dframe = pd.DataFrame(band_means, columns=bands_str, 
#                                  index=label_names)
#     output_dframe['AlphaPeak'] = alpha_peak
#     output_dframe.to_csv(output_filename, sep='\t')    
        
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root',
                        help='''Top level directory of the bids data'''
                        )
    parser.add_argument('-config', 
                        help='''Config file for processing data'''
                        )
    parser.add_argument('-subject',
                        help='''Subject to process'''
                        )
    parser.add_argument('-run',
                        help='''Run number.  \nNOTE: 01 is different from 1''',
                        default='1'
                        )
    parser.add_argument('-session',
                        default=None,
                        )
    parser.add_argument('-mains',
                        help='Electric mains frequency  (50 or 60)',
                        default=60.0,
                        )
    parser.add_argument('-rest_tag',
                        help='Override in case task name is other than rest\
                            for example - resteyesopen',
                        default='rest'
                        )
    parser.add_argument('-emptyroom_tag',
                        help='Override in case emptryoom is other than \
                            emptyroom',
                        default='emptyroom'
                        )
    parser.add_argument('-fs_ave_fids',
                        help='''If no fiducials have been localized to the mri
                        manually, this provides a coarse fit from the ave brain
                        which is fine tuned with the headshape.  This is less
                        acurate than a manually assessed fid placement''',
                        action='store_true',
                        default=False
                        )
                            
        
    args = parser.parse_args()
    
    proc = process(subject=args.subject, 
                bids_root=args.bids_root, 
                deriv_root=None,
                subjects_dir=None,
                rest_tagname=args.rest_tag,
                emptyroom_tagname=args.emptyroom_tag, 
                session=args.session, 
                mains=float(args.mains),
                run=args.run,
                t1_override=None,
                fs_ave_fids=args.fs_ave_fids
                )
    proc.load_data()
    proc.do_proc_allsteps()
    

    
