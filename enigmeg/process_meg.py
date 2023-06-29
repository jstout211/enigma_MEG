#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Jeff Stout and Allison Nugent
"""
import os
import os.path as op
import sys
import mne
from mne import Report
import numpy as np
import pandas as pd
import enigmeg
from enigmeg.spectral_peak_analysis import calc_spec_peak
from enigmeg.mod_label_extract import mod_source_estimate
from enigmeg.QA.enigma_QA_GUI_functions import build_status_dict
import logging
import munch 
import subprocess
import mne_bids
from mne_bids import get_head_mri_trans
from mne.beamformer import make_dics, apply_dics_csd
import scipy as sp
from mne_bids import BIDSPath
import functools
import MEGnet
from MEGnet.prep_inputs.ICA import main as ICA
from MEGnet.megnet_utilities import fPredictChunkAndVoting_parrallel
# Set tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow import keras



# define some variables

fmin = 1
fmax = 45
sfreq = 300
epoch_len = 4.0
mt_bandwidth = 2 # bandwidth for multitaper
n_bins = 177

logger=logging.getLogger()

# Function to retrieve the subject/session specific logger

def get_subj_logger(subjid, session, log_dir=None):
     '''Return the subject specific logger.
     This is particularly useful in the multiprocessing where logging is not
     necessarily in order'''
     fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
     sub_ses = f'{subjid}_ses_{session}'
     subj_logger = logging.getLogger(sub_ses)
     if subj_logger.handlers != []: # if not first time requested, use the file handler already defined
         tmp_ = [type(i) for i in subj_logger.handlers ]
         if logging.FileHandler in tmp_:
             return subj_logger
     else: # first time requested, add the file handler
         fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_ses-{session}_log.txt')
         fileHandle.setLevel(logging.INFO)
         fileHandle.setFormatter(logging.Formatter(fmt)) 
         subj_logger.addHandler(fileHandle)
         subj_logger.info('Initializing subject level enigma_anonymization log')
     return subj_logger   


#Decorator for logging functions

def log(function):
    def wrapper(*args, **kwargs):  
        logger.info(f"{function.__name__} :: START")
        output = function(*args, **kwargs)
        logger.info(f"{function.__name__} :: COMPLETED")
        return output
    return wrapper


# define a class that holds all the information about a single subject/session/run dataset for processing

class process():
    def __init__(
            self, 
            subject=None, 
            bids_root=None, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname='rest',
            emptyroom_tagname='emptyroom',
            session='1', 
            mains=60,
            run='1',
            t1_override=None,
            fs_ave_fids=False, 
            check_paths=True,
            csv_info=None
            ):
        
# =============================================================================
#         # Initialize variables and directories
# =============================================================================

        self.subject=subject.replace('sub-','')  # Strip sub- if present
        self.bids_root=bids_root
        if deriv_root is None:                   # Derivatives directory
            self.deriv_root = op.join(
                bids_root, 
                'derivatives'
                )
        else:
            self.deriv_root = deriv_root
            
        self.enigma_root = op.join( # enigma output directory
            self.deriv_root,
            'ENIGMA_MEG'
            )
        
        self.QA_dir = op.join( # QA output directory
            self.deriv_root,
            'ENIGMA_MEG_QA/sub-' + self.subject + '/ses-' + session 
            )
        
        if subjects_dir is None:    # Freesurfer subjects directory
            self.subjects_dir = op.join(
                self.deriv_root,
                'freesurfer',
                'subjects'
                )
        else:
            self.subjects_dir = subjects_dir
        self.proc_vars=munch.Munch()    # set some parameters
        self.proc_vars['fmin'] = fmin
        self.proc_vars['fmax'] = fmax
        self.proc_vars['sfreq'] = sfreq
        self.proc_vars['mains'] = mains
        self.proc_vars['epoch_len']=epoch_len  #seconds
        
        # t1_override is used when you are processing a single subject (so csv_info=None) but the 
        # name of the anatomical does not follow the same subj/ses/run hierarchy as the meg
        
        self._t1_override = t1_override   
        if fs_ave_fids==True:
            self._use_fsave_coreg=True
        else:
            self._use_fsave_coreg=False
        
        self._n_jobs = int(os.environ['n_jobs'])
            
# =============================================================================
#             Configure paths and filenames
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
        
        if emptyroom_tagname == None and not csv_info['eroom']:
            self.eroom_derivpath = None
        else:
            self.eroom_derivpath = self.deriv_path.copy().update(
                task=emptyroom_tagname,
                run=run
                )
        
        self.meg_rest_raw = self.bids_path.copy().update(
            datatype='meg', 
            task=rest_tagname, 
            run=run
            )
        
        if emptyroom_tagname == None and not csv_info['eroom']:
            self.meg_er_raw = None
        else:
            self.meg_er_raw = self.bids_path.copy().update(
                datatype='meg',
                task=emptyroom_tagname, 
                run=run
                )
        
        self.anat_bidspath = self.bids_path.copy().update(root=self.subjects_dir,
                                                          session=None,
                                                          check=False)
        
        if check_paths==True:
            self.check_paths()
            self.fnames=self.initialize_fnames(rest_tagname, emptyroom_tagname)
        elif csv_info is not None:
            self.fnames=self.initialize_fromcsv(csv_info)
            
        self.anat_vars=munch.Munch()
        self.anat_vars.fsdict = get_fs_filedict(self.subject,self.bids_root)
        self.anat_vars.process_list = compile_fs_process_list(self)              

            
    # This function initializes the bids path structures for the case where the paths have
    # been defined in a csv file, produced by parsing the BIDS tree. Note that the BidsPath
    # objects are specified separately for the MEG, eroom, and anat
            
    def initialize_fromcsv(self, csv_info):
        megpath=csv_info['path']            # get the filenames from the submitted csv row
        mripath=csv_info['mripath']
        eroompath=csv_info['eroom']
        datatype=csv_info['type']
        
        _tmp=munch.Munch()                  # initialize a temporary variable
        
        # extract the entities for the MEG and update the relevant BIDS path objectss
        entities = mne_bids.get_entities_from_fname(megpath)
        
        self.bids_path.update(
            session=entities['session'])
        self.deriv_path.update(
            session=entities['session'],
            extension=datatype)
        self.rest_derivpath.update(
            session=entities['session'],
            task=entities['task'],
            run=entities['run'],
            suffix='meg',
            extension='.fif')
        self.meg_rest_raw.update(
            session=entities['session'],
            task=entities['task'],
            run=entities['run'],
            suffix='meg',
            extension=datatype)
        self.QA_dir = BIDSPath(root=self.QA_dir,subject=subject, session=session)

        
        # if there's an emptyroom path provided, extract the entities from the filepath
        # and update the BIDS path objects
        
        if eroompath != None:
            entities = mne_bids.get_entities_from_fname(eroompath)
            self.eroom_derivpath.update(
                session=entities['session'],
                task=entities['task'],
                run=entities['run'],
                suffix='meg',
                extension='.fif')
            self.meg_er_raw.update(
                session=entities['session'],
                task=entities['task'],
                run=entities['run'],
                suffix='meg',
                extension=datatype)
            
        # Finally, extract the entities from the anatomical filename and update the 
        # BIDS path object
        
        entities = mne_bids.get_entities_from_fname(mripath)
        self.anat_bidspath.update(
                session=entities['session'],
                run=entities['run'])

        # check if the anatomical is .nii or .nii.gz
        
        _tmp['anat']=self.bids_path.copy().update(datatype='anat',extension='.nii')
        if not os.path.exists(_tmp['anat'].fpath):
            _tmp['anat']=self.bids_path.copy().update(datatype='anat',extension='.nii.gz')

        # populate the temporary dictionary _tmp with all the filenames
        
        _tmp['raw_rest']=self.meg_rest_raw
        if eroompath!=None:
            _tmp['raw_eroom']=self.meg_er_raw
        
        rest_deriv = self.rest_derivpath.copy().update(extension='.fif')
        if eroompath!=None:
            eroom_deriv = self.eroom_derivpath.copy().update(extension='.fif')
  
        _tmp['rest_filt']=rest_deriv.copy().update(processing='filt')
        if eroompath!=None:
            _tmp['eroom_filt']=eroom_deriv.copy().update(processing='filt')
        
        _tmp['rest_epo']=rest_deriv.copy().update(suffix='epo')
        if eroompath!=None:
            _tmp['eroom_epo']=eroom_deriv.copy().update(suffix='epo')
        
        _tmp['rest_csd']=rest_deriv.copy().update(suffix='csd', extension='.h5')
        if eroompath!=None:
            _tmp['eroom_csd']=eroom_deriv.copy().update(suffix='csd', extension='.h5')
             
        _tmp['rest_fwd']=rest_deriv.copy().update(suffix='fwd') 
        _tmp['rest_trans']=rest_deriv.copy().update(suffix='trans')
        _tmp['bem'] = self.deriv_path.copy().update(suffix='bem', extension='.fif')
        _tmp['src'] = self.deriv_path.copy().update(suffix='src', extension='.fif')
        
        _tmp['dics'] = self.deriv_path.copy().update(suffix='dics', 
                                                     run=self.meg_rest_raw.run,
                                                     extension='.h5')
        self.fooof_dir = self.deriv_path.directory / \
            f'sub-{self.subject}_ses-{self.meg_rest_raw.session}_fooof_results_run-{self.meg_rest_raw.run}'
        
        # Cast all bids paths to paths and save as dictionary
        path_dict = {key:str(i.fpath) for key,i in _tmp.items()}
        
        # Additional non-bids path files
        path_dict['parc'] = op.join(self.subjects_dir, 'morph-maps', 
                               f'sub-{self.subject}-fsaverage-morph.fif') 
        return munch.Munch(path_dict)
    
    # This function initializes all the filenames if a single subject has been requested
    # rather than a csv from a parsed BIDS tree
    
    def initialize_fnames(self, rest_tagname, emptyroom_tagname):
        '''Use the bids paths to generate output names'''
        _tmp=munch.Munch()
        rest_deriv = self.rest_derivpath.copy().update(extension='.fif')
        if emptyroom_tagname!=None:
            eroom_deriv = self.eroom_derivpath.copy().update(extension='.fif')
        
        ## Setup bids paths for all 
        # Conversion to actual paths at end
        
        _tmp['anat']=self.bids_path.copy().update(datatype='anat',extension='.nii')
        if not os.path.exists(_tmp['anat'].fpath):
            _tmp['anat']=self.bids_path.copy().update(datatype='anat',extension='.nii.gz')

        _tmp['raw_rest']=self.meg_rest_raw
        if emptyroom_tagname!=None:
            _tmp['raw_eroom']=self.meg_er_raw
        
        _tmp['rest_filt']=rest_deriv.copy().update(processing='filt')
        if emptyroom_tagname!=None:
            _tmp['eroom_filt']=eroom_deriv.copy().update(processing='filt')
        
        # MEGNET post
        #_tmp['rest_clean']=rest_deriv.copy().update(processing='clean')
        #_tmp['eroom_clean']=eroom_deriv.copy().update(processing='clean')
        
        _tmp['rest_epo']=rest_deriv.copy().update(suffix='epo')
        if emptyroom_tagname!=None:
            _tmp['eroom_epo']=eroom_deriv.copy().update(suffix='epo')
        
        #_tmp['rest_epo_clean']=_tmp['rest_epo'].copy().update(processing='clean')
        #_tmp['eroom_epo_clean']=_tmp['eroom_epo'].copy().update(processing='clean')
        
        _tmp['rest_csd']=rest_deriv.copy().update(suffix='csd', extension='.h5')
        if emptyroom_tagname!=None:
            _tmp['eroom_csd']=eroom_deriv.copy().update(suffix='csd', extension='.h5')
        
        _tmp['rest_fwd']=rest_deriv.copy().update(suffix='fwd') 
        _tmp['rest_trans']=rest_deriv.copy().update(suffix='trans')
        _tmp['bem'] = self.deriv_path.copy().update(suffix='bem', extension='.fif')
        _tmp['src'] = self.deriv_path.copy().update(suffix='src', extension='.fif')
        _tmp['ica'] = self.deriv_path.copy().update(suffix='ica', extension='.fif')
        _tmp['dics'] = self.deriv_path.copy().update(suffix='dics', 
                                                     run=self.meg_rest_raw.run,
                                                     extension='.h5')       
        self.fooof_dir = self.deriv_path.directory / \
            f'sub-{self.subject}_ses-{self.meg_rest_raw.session}_fooof_results_run-{self.meg_rest_raw.run}'
    
        # Cast all bids paths to paths and save as dictionary
        path_dict = {key:str(i.fpath) for key,i in _tmp.items()}
        
        # Additional non-bids path files
        path_dict['parc'] = op.join(self.subjects_dir, 'morph-maps', 
                               f'sub-{self.subject}-fsaverage-morph.fif') 
        return munch.Munch(path_dict)

# =============================================================================
#       Load data
# =============================================================================

    @log
    def load_data(self):
        if not hasattr(self, 'raw_rest'):
            self.raw_rest = load_data(self.meg_rest_raw.fpath) 
            self.raw_rest.pick_types(meg=True, eeg=False)
        if (not hasattr(self, 'raw_eroom')) and (self.meg_er_raw != None):
            self.raw_eroom = load_data(self.meg_er_raw.fpath) 
            self.raw_eroom.pick_types(meg=True, eeg=False)
        # figure out the MEG system vendor, note that this may be different from 
        # datatype if the datatype is .fif
        self.vendor = mne.channels.channels._get_meg_system(self.raw_rest.info)
    
    @log
    def check_paths(self):
        '''Verify that the raw data is present and can be found'''
        try:                            # Errors if MEG is not present
            self.meg_rest_raw.fpath  
            self.datatype = check_datatype(str(self.meg_rest_raw.fpath))
            
        except:
            logging.exception(f'Could not find rest dataset:{self.meg_rest_raw.fpath}\n')
            
        if hasattr(self, 'meg_er_raw'): 
            if self.meg_er_raw != None:
                try:                    # Error if empty room is specified but not present
                    self.meg_er_raw.fpath
                except:
                    logging.exception(f'Could not find emptyroom dataset:{self.meg_er_raw.fpath}\n')
            else:
                self.raw_eroom = None
        else:
            self.meg_er_raw = None
            self.raw_eroom = None
        
        # check if freesurfer directory is present for the subject
        subj_fsdir= op.join(f'{self.subjects_dir}', f'sub-{self.subject}')
        if not op.exists(subj_fsdir):
            errtxt = f'There is no freesurfer folder for {self.subject}:\
                              \n{subj_fsdir}'.replace('  ',' ')
            logging.exception(errtxt)

# =============================================================================
#       Vendor specific prep
# =============================================================================

    @log
    def vendor_prep(self):
        
        '''Different vendor types require special cleaning / initialization'''
        
        ## Apply 3rd order gradient for CTF datasets  
        if self.vendor[0] == 'CTF_275':
            if self.raw_rest.compensation_grade != 3:
                logging.info('Applying 3rd order gradient to rest data')
                self.apply_gradient_compensation(3)
            if hasattr(self, 'raw_eroom'):
                if self.raw_eroom.compensation_grade != 3:
                    logging.info('Applying 3rd order gradient to emptyroom data')
                    self.apply_gradient_compensation(3)
         
        # run bad channel assessments on rest and emptyroom (if present)
        rest_bad, rest_flat = assess_bads(self.meg_rest_raw.fpath, self.vendor[0])
        if hasattr(self, 'raw_eroom'):
            er_bad, er_flat = assess_bads(self.meg_er_raw.fpath, self.vendor[0], is_eroom=True)
        else:
            er_bad = []
            er_flat =[]
        all_bad = self.raw_rest.info['bads'] + self.raw_eroom.info['bads'] + \
                rest_bad + rest_flat + er_bad + er_flat
        # remove duplicates
        all_bad = list(set(all_bad))
            
        # mark bad/flat channels as such in datasets
        self.raw_rest.info['bads'] = all_bad
        if hasattr(self, 'raw_eroom'):
            self.raw_eroom.info['bads'] = all_bad
        
        print('bad or flat channels')
        print(all_bad)           
    
                   
# =============================================================================
#       Preprocessing
# =============================================================================

    @log
    def _preproc(self,          # resampling, mains notch filtering, bandpass filtering
                raw_inst=None,
                deriv_path=None):
        raw_inst.resample(self.proc_vars['sfreq'], n_jobs=self._n_jobs)
        raw_inst.notch_filter(self.proc_vars['mains'], n_jobs=self._n_jobs) 
        raw_inst.filter(self.proc_vars['fmin'], self.proc_vars['fmax'], n_jobs=self._n_jobs)
        raw_inst.save(deriv_path.copy().update(processing='filt', extension='.fif'), 
                      overwrite=True)
    
    @log
    def do_ica(self):           # perform the 20 component ICA using functions from megnet
        ica_basename = self.meg_rest_raw.basename + '_ica'
        ICA(self.fnames['raw_rest'],mains_freq=self.proc_vars['mains'], save_preproc=True, save_ica=True, 
        results_dir=self.deriv_path.directory, outbasename=ica_basename)  
        self.fnames.ica_folder = self.deriv_path.directory  / ica_basename
        self.fnames.ica = self.fnames.ica_folder / (ica_basename + '_0-ica.fif')
        self.fnames.ica_megnet_raw =self.fnames.ica_folder / (ica_basename + '_250srate_meg.fif')

    def prep_ica_qa(self):      # if desired, create QA images for ICA components
        ica_fname = self.fnames.ica
        raw_fname = self.fnames.ica_megnet_raw
        
        prep_fcn_path = op.join(enigmeg.__path__[0], 'QA/make_ica_qa.py')
        
        output_path = self.bids_root + '/derivatives/ENIGMA_MEG_QA/sub-' + self.subject + '/ses-' + self.meg_rest_raw.session
        
        subprocess.run(['python', prep_fcn_path, '-ica_fname', ica_fname, '-raw_fname', raw_fname,
                        '-vendor', self.vendor[0], '-results_dir', output_path, '-basename', self.meg_rest_raw.basename])
    @log           
    def do_classify_ica(self):  # use the MEGNET model to automatically classify ICA components as artifactual
        import tensorflow_addons as tfa #Required for loading. tfa f1_score embedded in model
        from scipy.io import loadmat
        model_path = op.join(MEGnet.__path__[0] ,  'model/MEGnet_final_model.h5')
        # This is set to use CPU in initial import
        kModel=keras.models.load_model(model_path)
        arrSP_fnames = [op.join(self.fnames.ica_folder, f'component{i}.mat') for i in range(1,21)]
        arrTS = loadmat(op.join(self.fnames.ica_folder, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
        arrSP = np.stack([loadmat(i)['array'] for i in arrSP_fnames])
        preds, probs = fPredictChunkAndVoting_parrallel(kModel, arrTS, arrSP)
        self.meg_rest_ica_classes = preds.argmax(axis=1)
        self.ica_comps_toremove = [index for index, value in enumerate(self.meg_rest_ica_classes) if value in [1, 2, 3]]
        logstring = 'MEGNET classifications: ' + str(self.meg_rest_ica_classes)
        logger.info(logstring)
        logstring = 'Components to reject: ' + str(self.ica_comps_toremove)
        logger.info(logstring)

    def set_ica_comps_manual(self): # If components were selected using manual QA, parse log files to generate list
        newdict = parse_manual_ica_qa(self)
        self.ica_comps_toremove = np.asarray(newdict[self.meg_rest_raw.basename]).astype(int)
        logstring = 'Components to reject: ' + str(self.meg_rest_raw.icacomps)
        logger.info(logstring)
        
    @log
    def do_clean_ica(self):         # Remove identified ICA components
        print("removing ica components")
        ica=mne.preprocessing.read_ica(op.join(self.ica_dir,self.ica_fname))
        ica.exclude = self.meg_rest_raw.icacomps
        self.load_data()
        ica.apply(self.raw_rest)
        
    @log
    def do_preproc(self):           # proc both rest and empty room
        '''Preprocess both datasets'''
        self._preproc(raw_inst=self.raw_rest, deriv_path=self.rest_derivpath)
        if self.raw_eroom != None:
            self._preproc(raw_inst=self.raw_eroom, deriv_path=self.eroom_derivpath)
    
    @log
    def _proc_epochs(self,          # divide the rest data into epochs
                     raw_inst=None,
                     deriv_path=None):
        '''Create and save epochs
        Create and save cross-spectral density'''
        epochs = mne.make_fixed_length_epochs(raw_inst, 
                                              duration=self.proc_vars['epoch_len'], 
                                              preload=True)
        epochs_fname = deriv_path.copy().update(suffix='epo', extension='.fif')
        epochs.save(epochs_fname, overwrite=True)
        
        # compute the cross spectral density for the epoched data
        # multitaper better but slower, use fourier for testing
        
        #csd = mne.time_frequency.csd_fourier(epochs,fmin=fmin,fmax=fmax,n_jobs=self._n_jobs)
        csd = mne.time_frequency.csd_multitaper(epochs,fmin=fmin,fmax=fmax,n_jobs=self._n_jobs)
        csd_fname = deriv_path.copy().update(suffix='csd', extension='.h5')
        csd.save(str(csd_fname.fpath), overwrite=True)
    
    @log
    def do_proc_epochs(self):   # epoch both the rest and the empty room
        self._proc_epochs(raw_inst=self.raw_rest,
                          deriv_path=self.rest_derivpath)
        if self.raw_eroom != None:
            self._proc_epochs(raw_inst=self.raw_eroom, 
                              deriv_path=self.eroom_derivpath)
    
    @log                        # Process the anatomical MRI
    def proc_mri(self, t1_override=None,redo_all=False):
        
        # if not provided with a separate T1 MRI filename, extract it from the BIDSpath objects
        if t1_override is not None:
            entities=mne_bids.get_entities_from_fname(t1_override)
            t1_bids_path = BIDSPath(**entities)
        else:
            t1_bids_path = self.bids_path.copy().update(datatype='anat', 
                                                    suffix='T1w')
                                                    
        # Configure filepaths
        os.environ['SUBJECTS_DIR']=self.subjects_dir
        deriv_path = self.rest_derivpath.copy()
        
        fs_subject = 'sub-' + self.subject
        
        fs_subject_dir = os.path.join(self.subjects_dir,fs_subject)
        
        # check to see if Freesurfer is running (or if it crashed out)
        if 'IsRunning.lh+rh' in os.listdir(os.path.join(fs_subject_dir, 'scripts')):
            del_run_file=input('''The IsRunning.lh+rh file is present.  Could be from a broken process \
                  or the process is currently running.  Do you want to delete to continue?(y/n)''')
            if del_run_file.lower()=='y':
                os.remove(os.path.join(fs_subject_dir, 'scripts','IsRunning.lh+rh'))
                
        # do all the remaining freesurfer processing steps        
        for proc in self.anat_vars.process_list:
            subcommand(proc)
        
        # Update path so src/bem are not saved as task/run specific
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif',
                                             task=None, run=None)
        bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif',
                                             task=None, run=None)
        
        # Specific to task = rest / run = run#
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')

        # check to see if stuff is there, and if it isn't, make it                
        if fwd_fname.fpath.exists() and (redo_all is False):        # forward solution
            fwd = mne.read_forward_solution(fwd_fname)
            self.rest_fwd = fwd
        
        watershed_check_path=op.join(self.subjects_dir,             # watershed bem
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
            
        if (not src_fname.fpath.exists()) or (redo_all is True):    # source space
            src = mne.setup_source_space(f'sub-{self.subject}',
                                         spacing='oct6', add_dist='patch',
                                 subjects_dir=self.subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
        
        if (not trans_fname.fpath.exists()) or (redo_all is True):  # transformation
            if self._use_fsave_coreg==True:
                trans = self.do_auto_coreg()
            else:
                if(self.datatype == 'fif'):
                    trans = get_head_mri_trans(self.meg_rest_raw, {'allow_maxshield' : True},
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject) 
                elif(self.datatype == 'ctf'):
                    trans = get_head_mri_trans(self.meg_rest_raw, {'system_clock' : 'ignore'},
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject)
                else:
                    trans = get_head_mri_trans(self.meg_rest_raw,
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject)
            mne.write_trans(trans_fname.fpath, trans, overwrite=True)
        else:
            trans = mne.read_trans(trans_fname.fpath)
        # make the forward solution
        fwd = mne.make_forward_solution(self.raw_rest.info, trans, src, bem_sol, eeg=False, 
                                        n_jobs=self._n_jobs)
        self.rest_fwd=fwd
        mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)
    
    @log
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
            
    # make the parcellation and subparcellation
    @log
    def do_make_aparc_sub(self):
        write_aparc_sub(subjid=f'sub-{self.subject}', 
                        subjects_dir=self.subjects_dir)
        
# =============================================================================
#       Source Localization and spectral parameterization
# =============================================================================
         
    @log
    def do_beamformer(self):    # Function to do the beamformer    
        
        # read in all the necessary files
        dat_csd = mne.time_frequency.read_csd(self.fnames.rest_csd)
        forward = mne.read_forward_solution(self.fnames.rest_fwd)
        epochs = mne.read_epochs(self.fnames.rest_epo)
        fname_dics = self.fnames.dics #Pre-assign output name
        
        # If emptyroom present - use in beamformer
        if self.meg_er_raw != None:
            noise_csd = mne.time_frequency.read_csd(self.fnames.eroom_csd)
            noise_rank = mne.compute_rank(self.raw_eroom)
            epo_rank = mne.compute_rank(epochs)
            if 'mag' in epo_rank:
                if epo_rank['mag'] < noise_rank['mag']:
                    noise_rank['mag']=epo_rank['mag']
            if 'grad' in epo_rank:
                if epo_rank['grad'] < noise_rank['grad']:
                    noise_rank['grad']=epo_rank['grad']
            filters=mne.beamformer.make_dics(epochs.info, forward, dat_csd, reg=0.05, pick_ori='max-power',
                    noise_csd=noise_csd, inversion='matrix', weight_norm='unit-noise-gain', rank=noise_rank)

        else:
            #Build beamformer without emptyroom noise
            epo_rank = mne.compute_rank(epochs)
            filters=mne.beamformer.make_dics(epochs.info, forward, dat_csd, reg=0.05,pick_ori='max-power',
                    inversion='matrix', weight_norm='unit-noise-gain', rank=noise_rank)
        
        filters.save(fname_dics, overwrite=True)
        psds, freqs = apply_dics_csd(dat_csd, filters) 
        self.psds = psds
        self.freqs = freqs
        
    @log    
    def do_label_psds(self):    # Function to extract the psd for each label. Takes a long time. 
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
        
        label_ts = mne.source_estimate.extract_label_time_course(self.psds, 
                                                                 labels, 
                                                                 self.rest_fwd['src'],
                                                                 mode='pca')

        #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
        self.label_ts = np.stack(label_ts)
    
    @log    
    def do_spectral_parameterization(self):
        '''
        Passes spectra to fooof alg for peak and 1/f analysis

        Returns
        -------
        None.

        '''
        freq_bins = np.array(self.freqs)    
    
        #Initialize 
        labels = self.labels
        label_power = np.zeros([len(labels), len(freq_bins)])  
        alpha_peak = np.zeros(len(labels))
        
        outfolder = self.deriv_path.directory / \
            f'sub-{self.subject}_ses-{self.meg_rest_raw.session}_fooof_results_run-{self.meg_rest_raw.run}'
        self.results_dir = outfolder
        if not os.path.exists(outfolder): os.mkdir(outfolder)
        
        #Create PSD for each label
        label_stack = self.label_ts
        for label_idx in range(len(self.labels)):
            #current_psd = label_stack[:,label_idx, :].mean(axis=0)  ###???
            current_psd = label_stack[label_idx, :]
            label_power[label_idx,:] = current_psd
            
            #spectral_image_path = os.path.join(outfolder, 'Spectra_'+
            #                                    labels[label_idx].name + '.png')   
            spectral_image_path = None  ## supress output of spectra .png files for every region
    
            try:
                tmp_fmodel = calc_spec_peak(freq_bins, current_psd, 
                                out_image_path=spectral_image_path)
                
                # work around for when fooof identifies multiple alpha peaks - set to np.nan
                potential_alpha_idx = np.where((8.0 <= tmp_fmodel.peak_params[:,0] ) & \
                                        (tmp_fmodel.peak_params[:,0] <= 12.0 ) )[0]
                if len(potential_alpha_idx) != 1:
                    alpha_peak[label_idx] = np.nan         
                else:
                    alpha_peak[label_idx] = tmp_fmodel.peak_params[potential_alpha_idx[0]][0]
                    print('label_idx: %d, alpha_peak: %f' % (label_idx, alpha_peak[label_idx]))
            except:
                alpha_peak[label_idx] = np.nan  # case where no alpha peak is identified - set to np.nan
            
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

        # output some freesurfer QA metrics

        subcommand(f'mri_segstats --qa-stats sub-{self.subject} {self.enigma_root}/sub-{self.subject}/ses-{self.meg_rest_raw.session}/sub-{self.subject}_fsstats.tsv')          
        
# =============================================================================
#       Perform all functions on an instance of the process class
# =============================================================================
    
    def do_proc_allsteps(self):             # do all the steps for single subject command line processing
        self.load_data()
        self.vendor_prep()
        self.do_ica()
        self.do_classify_ica()
        self.do_preproc()
        self.do_clean_ica()
        self.do_proc_epochs()
        self.proc_mri(t1_override=self._t1_override)
        self.do_beamformer()
        self.do_make_aparc_sub()
        self.do_label_psds()
        self.do_spectral_parameterization()
        
# =============================================================================
#       Super secret hidden debugging functions. 
# =============================================================================
        
    # hidden debugging function to list all output files generated
    def list_outputs(self):
        exists = [i for i in self.fnames if op.exists(self.fnames[i])]
        missing = [i for i in self.fnames if not op.exists(self.fnames[i])]
        for i in exists:
            print(f'Present: {self.fnames[i]}')
        print('\n\n')
        for i in missing:
            print(f'Missing: {self.fnames[i]}')  
    
    # hidden debugging function to check alignment
    def check_alignment(self): # test function to look at alignment - not called
        self.trans = mne.read_trans(self.fnames['rest_trans'])
        self.load_data()
        mne.viz.plot_alignment(self.raw_rest.info,
                                    trans=self.trans,
                                    subject='sub-'+self.bids_path.subject,
                                    subjects_dir=self.subjects_dir,
                                    dig=True)
 
# =========================== End of class and method =========================        
        
# =============================================================================
#       Utility and helper functions
# =============================================================================
    
def subcommand(function_str):               # simple function to run something on the command line
    from subprocess import check_call
    check_call(function_str.split(' '))
        
def compile_fs_process_list(process):       # function to determine what freesurfer steps still must be run
    process_steps=[]
    fs_subject = 'sub-'+process.subject
    if not(process.anat_vars.fsdict['001mgz']):
        process_steps.append('recon-all -i {} -s {}'.format(process.fnames['anat'], fs_subject))
    if not(process.anat_vars.fsdict['brainmask']):
        process_steps.append('recon-all -autorecon1 -s {}'.format(fs_subject))
    if not(process.anat_vars.fsdict['lh_pial']):
        process_steps.append('recon-all -autorecon2 -careg -s {}'.format(fs_subject))
    if not(process.anat_vars.fsdict['lh_dkaparc']):
        process_steps.append('recon-all -autorecon3 -s {}'.format(fs_subject))
    return process_steps   
        

def get_fs_filedict(subject, bids_root):    # make a dictionary of freesurfer filenames
    subjects_dir = op.join(bids_root, 'derivatives', 'freesurfer', 'subjects')
    
    fs_dict = {}
    # FS default files
    fs_dict['001mgz'] = f'sub-{subject}/mri/orig/001.mgz'
    fs_dict['brainmask'] = f'sub-{subject}/mri/brainmask.mgz'
    fs_dict['lh_pial'] = f'sub-{subject}/surf/lh.pial'
    fs_dict['rh_pial'] = f'sub-{subject}/surf/rh.pial'
    fs_dict['lh_dkaparc'] = f'sub-{subject}/label/lh.aparc.DKTatlas.annot'
    fs_dict['rh_dkaparc'] = f'sub-{subject}/label/rh.aparc.DKTatlas.annot'
    
    # FS - post process files
    fs_dict['in_skull'] = f'sub-{subject}/bem/inner_skull.surf'
    fs_dict['head'] = f'sub-{subject}/bem/sub-{subject}-head.fif'
    fs_dict['head_dense'] = f'sub-{subject}/bem/sub-{subject}-head-dense.fif'
    
    # Morph maps for sub-aparc
    fs_dict['morph'] = f'morph-maps/fsaverage-sub-{subject}-morph.fif'
    
    # Prepend subjects_dir
    for key, value in fs_dict.items():
        fs_dict[key] = op.join(subjects_dir,value)
        if not op.exists(fs_dict[key]):
            fs_dict[key]=False
    return fs_dict
 
def check_datatype(filename):               # function to determine the file format of MEG data 
    '''Check datatype based on the vendor naming convention to choose best loader'''
    if os.path.splitext(filename)[-1] == '.ds':
        return 'ctf'
    elif os.path.splitext(filename)[-1] == '.fif':
        return 'fif'
    elif os.path.splitext(filename)[-1] == '.4d' or ',' in filename:
        return '4d'
    elif os.path.splitext(filename)[-1] == '.sqd':
        return 'kit'
    elif os.path.splitext(filename)[-1] == 'con':
        return 'kit'
    else:
        raise ValueError('Could not detect datatype')
        
def return_dataloader(datatype):            # function to return a data loader based on file format
    '''Return the dataset loader for this dataset'''
    if datatype == 'ctf':
        return functools.partial(mne.io.read_raw_ctf, system_clock='ignore')
    if datatype == 'fif':
        return functools.partial(mne.io.read_raw_fif, allow_maxshield=True)
    if datatype == '4d':
        return mne.io.read_raw_bti
    if datatype == 'kit':
        return mne.io.read_raw_kit

def load_data(filename):                    # simple function to load raw MEG data
    datatype = check_datatype(filename)
    dataloader = return_dataloader(datatype)
    raw = dataloader(filename, preload=True)
    return raw

def assess_bads(raw_fname, vendor, is_eroom=False): # assess MEG data for bad channels
    '''Code sampled from MNE python website
    https://mne.tools/dev/auto_tutorials/preprocessing/\
        plot_60_maxwell_filtering_sss.html'''
    from mne.preprocessing import find_bad_channels_maxwell
    # load data with load_data to ensure correct function is chosen
    raw = load_data(raw_fname)    
    if raw.times[-1] > 60.0:
        raw.crop(tmax=60)    
    raw.info['bads'] = []
    raw_check = raw.copy()
    
    if vendor == '306m' or vendor == '122m':
        
        if is_eroom==False:
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, cross_talk=None, calibration=None,
                return_scores=True, verbose=True)
        else:
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, cross_talk=None, calibration=None,
                return_scores=True, verbose=True, coord_frame="meg")
            
        # find_bad_channels_maxwell is actually pretty bad at finding flat channels - 
        # it uses a much too stringent threshold. So, we need some supplementary code
        # This is extra complicated for Elekta/MEGIN, because there are both mags and 
        # grads, which will be on a different scale
            
        mags = mne.pick_types(raw_check.info, meg='mag')
        grads = mne.pick_types(raw_check.info, meg='grad')
        # get the standard deviation for each channel, and the trimmed mean of the stds
        # have to do this separately for mags and grads
        stdraw_mags = np.std(raw_check._data[mags,:],axis=1)
        stdraw_grads = np.std(raw_check._data[grads,:],axis=1)    
        stdraw_trimmedmean_mags = sp.stats.trim_mean(stdraw_mags,0.1)
        stdraw_trimmedmean_grads = sp.stats.trim_mean(stdraw_grads,0.1)
        # we can't use the same threshold here, because grads have a much greater 
        # variance in the variances 
        flat_mags = np.where(stdraw_mags < stdraw_trimmedmean_mags/100)[0]
        flat_grads = np.where(stdraw_grads < stdraw_trimmedmean_grads/1000)[0]
        # need to use list comprehensions
        flat_idx_mags = [flat_mags[i] for i in flat_mags.tolist()]
        flat_idx_grads = [flat_grads[i] for i in flat_grads.tolist()]
        flats = []
        for flat in flat_idx_mags:
            flats.append(raw_check.info['ch_names'][mags[flat_idx_mags]])
        for flat in flat_idx_grads:
            flats.append(raw_check.info['ch_names'][grads[flat_idx_grads]])
        
    # ignore references and use 'meg' coordinate frame for CTF and KIT
    
    if vendor == 'CTF_275':
        raw_check.apply_gradient_compensation(0)
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None, coord_frame='meg',
            return_scores=True, verbose=True, ignore_ref=True)
        
        # again, finding flat/bad channels is not great, so we add another algorithm
        # since other vendors don't mix grads and mags, we only need to do this for
        # a single channel type
        
        megs = mne.pick_types(raw_check.info, meg=True)
        # get the standard deviation for each channel, and the trimmed mean of the stds
        stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
        stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
        flat_megs = np.where(stdraw_megs < stdraw_trimmedmean_megs/100)[0]
        # need to use list comprehensions
        flat_idx_megs = [flat_megs[i] for i in flat_megs.tolist()]
        flats = []
        for flat in flat_idx_megs:
            flats.append(raw_check.info['ch_names'][megs[flat_idx_mags]]) 
    
    else: 
        
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None, coord_frame='meg',
            return_scores=True, verbose=True, ignore_ref=True)
    
        # again, finding flat/bad channels is not great, so we add another algorithm
        # since other vendors don't mix grads and mags, we only need to do this for
        # a single channel type
    
        megs = mne.pick_types(raw_check.info, meg=True)
        # get the standard deviation for each channel, and the trimmed mean of the stds
        stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
        stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
        flat_megs = np.where(stdraw_megs < stdraw_trimmedmean_megs/100)[0]
        # need to use list comprehensions
        flat_idx_megs = [flat_megs[i] for i in flat_megs.tolist()]
        flats = []
        for flat in flat_idx_megs:
            flats.append(raw_check.info['ch_names'][megs[flat_idx_mags]])    
    
    auto_flat_chs = auto_flat_chs + flats
    auto_flat_chs = list(set(auto_flat_chs))
            
    return auto_noisy_chs, auto_flat_chs            

def write_aparc_sub(subjid=None, subjects_dir=None):    # write the parcel annotation, fetch fsaverage if needed
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

# hidden testing functions
def load_test_data(**kwargs):
    os.environ['n_jobs']='6'
    proc = process(subject='ON02747',
                        bids_root=op.expanduser('~/ds004215'),
                        session='01',
                        run='01',
                        emptyroom_tagname='noise', 
                        mains=60,
                        t1_override='~/ds004215/sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
                        **kwargs)
    # proc.load_data()
    return proc

# hidden testing functions
def load_test_data_noer(**kwargs):
    os.environ['n_jobs']='6'
    proc = process(subject='ON02747',
                        bids_root=op.expanduser('~/ds004215'),
                        session='01',
                        emptyroom_tagname=None, 
                        mains=60,
                        t1_override='~/ds004215/sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
                        **kwargs)
    # proc.load_data()
    return proc

# print(__name__)
# # For Testing run this cell
# if __name__!='__main__':
# proc = load_test_data()

# simple function to get frequency indices
def get_freq_idx(bands, freq_bins):
    ''' Get the frequency indexes'''
    output=[]
    for band in bands:
        tmp = np.nonzero((band[0] < freq_bins) & (freq_bins < band[1]))[0]   ### <<<<<<<<<<<<< Should this be =<...
        output.append(tmp)
    return output

# =============================================================================
#       Master processing functions 
# =============================================================================

def process_subject(subject, args):
    logger = get_subj_logger(subject, args.session, log_dir)
    logger.info('Initializing structure')
    proc = process(subject=subject, 
            bids_root=args.bids_root, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname=args.rest_tag,
            emptyroom_tagname=args.emptyroom_tag, 
            session=args.session, 
            mains=float(args.mains),
            run=args.run,
            t1_override=None,
            fs_ave_fids=args.fs_ave_fids)           
    proc.do_proc_allsteps()
    
def process_subject_up_to_icaqa(subject, args):
    logger = get_subj_logger(subject, args.session, log_dir)
    logger.info('Initializing structure')
    proc = process(subject=subject, 
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
    proc.do_ica()
    proc.prep_ica_qa()    
    
def process_subject_after_icaqa(subject, args):
    logger = get_subj_logger(subject, args.session, log_dir)
    logger.info('Initializing structure')
    proc = process(subject=subject, 
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
    proc.set_ica_comps_manual()
    proc.do_preproc()
    proc.do_clean_ica()
    proc.proc_mri(t1_override=proc._t1_override)
    proc.do_beamformer()
    proc.do_make_aparc_sub()
    proc.do_label_psds()
    proc.do_spectral_parameterization()
        
def parse_manual_ica_qa(self):
    logfile_path = self.bids_root + '/derivatives/ENIGMA_MEG_QA/ica_QA_logfile.txt'
    with open(logfile_path) as f:
        logcontents = f.readlines()
    dictionary = build_status_dict(logcontents)

    newdict = {}
    for key, value in dictionary.items():
        subjrun = key.split('_icacomp-')[0]

        if newdict == {}:  # if this is the first key and the new dict is empty
            if dictionary[key].strip('\n') == 'BAD':  # if component is bad
                dropcomp = key.split('_icacomp-')[1].split('.png')[0]
                newdict = {subjrun: [dropcomp]}
            else:
                newdict = {subjrun: []}  # if compoenent is good
                   
        elif subjrun in newdict.keys():   # If the key is already in the new dict
            if dictionary[key].strip('\n') == 'BAD':  # If component is bad (if comp is good, do nothing)
                dropcomp = key.split('_icacomp-')[1].split('.png')[0]
                newdict[subjrun].append(dropcomp)
    
        else: # if the key isn't in the new dictionary
            if dictionary[key].strip('\n') == 'BAD':  # if compoenent is bad
                dropcomp = key.split('_icacomp-')[1].split('.png')[0]
                newdict[subjrun] = [dropcomp]
            else:
                newdict[subjrun] = [] # if compoenent is good
    return newdict

#%%    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root',
                        help='''Top level directory of the bids data'''
                        )
    # parser.add_argument('-config', 
    #                     help='''Config file for processing data'''
    #                     )
    parser.add_argument('-subject',
                        help='''BIDS ID of subject to process''',
                        default=None
                        )
    parser.add_argument('-subjects_dir',
                        help='''Freesurfer subjects directory, only specify if not \
                        bids_root/derivatives/freesurfer/subjects'''
                        )
    parser.add_argument('-fs_subject',
                        help='''Freefurfer subject ID if different from BIDS ID'''
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
                            emptyroom.  In case of no emptyroom, set as None on cmdline',
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
    parser.add_argument('-proc_fromcsv',
                        help='''Loop over all subjects in the bids_root
                        and process. Requires CSV file with processing manifest''',
                        default=None
                        )
    parser.add_argument('-n_jobs',
                        help='''number of jobs to run concurrently for 
                        multithreaded operations''',
                        default=1
                        )
    parser.add_argument('-ica_manual_qa_prep',
                        help='''if flag is present, stop after ICA for manual QA''',
                        action='store_true',
                        default=0
                        )
    parser.add_argument('-process_manual_ica_qa',
                        help='''If flag is present, pick up analysis after performing manual ICA QA''',
                        action='store_true',
                        default=0
                        )
                                   
    args = parser.parse_args()
    
    logger=logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    
    n_jobs = args.n_jobs  #extract this from the configuration file
    os.environ['n_jobs'] = str(n_jobs)
    
    # print help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(1) 
    
    # set some defaults
    if args.run:
        if args.run.lower()=='none': args.run=None
    if args.session:
        if args.session.lower()=='none': args.session=None
    if args.emptyroom_tag:
        if args.emptyroom_tag.lower()=='none': args.emptyroom_tag=None
    
    if not args.bids_root:
        bids_root = op.join(os.getcwd(), 'bids_out')
        args.bids_root = bids_root
    else:
        bids_root=args.bids_root
        
    if not op.exists(bids_root):    # throw an error if the BIDS root directory doesn't exist
        parser.print_help()
        raise ValueError('Please specify a correct -bids_root')     
    
    # To make file handling easier, even if there is another subjects directory, we'll create one in 
    # the BIDS derivatives/ folder and set up symbolic links there. 
    
    if not os.path.isdir(os.path.join(bids_root,'derivatives')):
        os.makedirs(os.path.join(bids_root,'derivatives'))
    if not os.path.isdir(os.path.join(bids_root,'derivatives/freesurfer')):
        os.makedirs(os.path.join(bids_root,'derivatives/freesurfer'))
    if not os.path.isdir(os.path.join(bids_root,'derivatives/freesurfer/subjects')):
        os.makedirs(os.path.join(bids_root,'derivatives/freesurfer/subjects'))
        
    # We have to find out if there is an fsaverage in the freesurfer directory, and if there is, if 
    # it is a sym link. If it's a link, it will break later when we try to get the bem directory
    
    if os.path.isdir(os.path.join(bids_root,'derivatives/freesurfer/subjects/fsaverage')):
        if os.path.islink(os.path.join(bids_root,'derivatives/freesurfer/subjects/fsaverage')):
            raise ValueError('$SUBJECTS_DIR/fsaverage cannot be a symlink; remove and rerun process_MEG to fetch data')
            
    # check and make sure all fsaverage files are present and download if not. 
    mne.datasets.fetch_fsaverage(op.join(bids_root,'derivatives/freesurfer/subjects/'))
    
    log_dir = f'{bids_root}/derivatives/ENIGMA_MEG/logs'
    if not os.path.isdir(os.path.join(bids_root,'derivatives/ENIGMA_MEG')):
        os.makedirs(os.path.join(bids_root,'derivatives/ENIGMA_MEG'))
    if not os.path.isdir(os.path.join(bids_root,'derivatives/ENIGMA_MEG/logs')):
        os.makedirs(os.path.join(bids_root,'derivatives/ENIGMA_MEG/logs'))
        
    # single subject vs. multiple subject processing.     
    
    if args.subject:    
        
        args.subject=args.subject.replace('sub-','') # strip the sub- off for uniformity
        print(args.subject)
        
        if args.proc_fromcsv != None:
            raise ValueError("You can't specify both a subject id and a csv file, sorry")
            
        print('processing a single subject %s' % args.subject)      
            
        # now, parse the inputs for freesurfer directory and subject (if not in derivatives)
        # so freesurfer processing doesn't have to be repeated if it exists elsewhere
    
        if args.subjects_dir:
            
            dir_entered = os.path.abspath(args.subjects_dir) # get absolute paths to compare
            default_dir = os.path.abspath(os.path.join(bids_root, 'derivatives/freesurfer/subjects'))
            
            if dir_entered == default_dir: # don't use -subjects_dir if using the default subjects_dir                
                print('Specified FS subjects dir same as default')
                
            elif args.fs_subject:
                if args.fs_subject == ('sub-' + args.subject):
                    raise ValueError('Specified FS subject ID is same as subject ID, please remove -fs_subject and try again')
                else:
                    # make a symbolic link from the existing freesurfer directory to the subjects directory 
                    # in the derivatives folder in the bids tree. Make sure we match the bids name
                    print('linking freesurfer subject %s to bids subject %s in derivatives folder',(args.fs_subject, args.subject))
                    subprocess.call(['ln','-s',os.path.join(dir_entered, args.fs_subject),
                                 os.path.join(default_dir, 'sub-'+args.subject)])
                    
            else: # case where a different subjects dir was entered, but not a separate fs ID
                # make sure subjects directory exists with same ID as BIDS ID
                # make a symbolic link for the subject in the derivatives/freesurfer/subjects directory
                if os.path.isdir(os.path.join(dir_entered, 'sub-'+args.subject)):
                        subprocess.call(['ln','-s',os.path.join(dir_entered, 'sub-'+args.subject),
                            os.path.join(default_dir, 'sub-'+args.subject)])
                else: 
                    raise ValueError('No folder for subject in specified subjects_dir, please try again')
            
            # now that we've set up the symbolic links, we can now use the default subjects directory
            args.subjects_dir = default_dir
      
        logger = get_subj_logger(args.subject, args.session, log_dir)
        logger.info(f'processing subject {args.subject} session {args.session}')
        
        if args.ica_manual_qa_prep:
            
            qa_dir = f'{bids_root}/derivatives/ENIGMA_MEG_QA'
            if not os.path.isdir(qa_dir):
                os.makedirs(qa_dir)
            if not os.path.isdir(os.path.join(qa_dir,'sub-'+args.subject)):
                os.makedirs(os.path.join(qa_dir,'sub-'+args.subject))
            if not os.path.isdir(os.path.join(qa_dir,'sub-'+args.subject+'/ses-'+args.session)):
                os.makedirs(os.path.join(qa_dir,'sub-'+args.subject+'/ses-'+args.session))
            process_subject_up_to_icaqa(args.subject, args)

        elif args.process_manual_ica_qa:
            process_subject_after_icaqa(args.subject, args)

        else:
            process_subject(args.subject, args)  # process the single specified subject
  
    ## batch processing from a .csv file
               
    elif args.proc_fromcsv:
        
        # if the user specified a subjects_dir, check to see if it is the same as the default dir
        if args.subjects_dir:
                
            dir_entered = os.path.abspath(args.subjects_dir) # get absolute paths to compare
            default_dir = os.path.abspath(os.path.join(bids_root, 'derivatives/freesurfer/subjects'))
            
            if dir_entered == default_dir: # if the directories are the same, replace with absolute path
                print('Specified FS subjects dir same as default')
                args.subjects_dir = default_dir
                need_link = False
            else:
                args.subjects_dir = dir_entered
                need_link = True
        
        print('processing subject list from %s' % args.proc_fromcsv)
        
        dframe = pd.read_csv(args.proc_fromcsv, dtype={'sub':str, 'run':str, 'ses':str})
        dframe = dframe.astype(object).replace(np.nan,None)
        
        for idx, row in dframe.iterrows():  # iterate over each row in the .csv file
            
            print(row)
            
            subject=row['sub']
            subject = subject.replace('sub-','')
            
            session=str(row['ses'])
            logger = get_subj_logger(subject, session, log_dir)
            logger.info(f'processing subject {subject} session {session}')
            

            if row['mripath'] == None:
                logger.info('No MRI, cannot process any further')
                print("Can't process subject %s, no MRI found" % args.subject)
            
            else:             
                process_subj = process(subject = subject,
                                       bids_root = bids_root,
                                       deriv_root = None,
                                       subjects_dir = args.subjects_dir,
                                       rest_tagname = None,
                                       emptyroom_tagname = None,
                                       session = session,
                                       mains = args.mains,
                                       run = str(row['run']),
                                       t1_override=None,
                                       fs_ave_fids=False,
                                       check_paths=False,
                                       csv_info=row)
                process_subj.load_data()
                
                if (args.ica_manual_qa_prep == 1):
                    process_subj.do_ica()
                    process_subj.prep_ica_qa()
                elif(args.process_manual_ica_qa == 1):
                    process_subj.set_ica_comps_manual()
                    process_subj.do_preproc()
                    process_subj.do_clean_ica()
                    process_subj.proc_mri()
                    process_subj.do_beamformer()
                    process_subj.do_make_aparc_sub()
                    process_subj.do_label_psds()
                    process_subj.do_spectral_parameterization()
                    
                else:    
                    process_subj.do_proc_allsteps()