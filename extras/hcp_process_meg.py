#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:29:34 2024

@author: nugenta
"""
import os
import os.path as op
import sys
import mne
import logging
import munch 
import subprocess
import mne_bids

from enigmeg.process_meg import process 
from enigmeg.process_meg import subcommand
from enigmeg.process_meg import log, get_subj_logger

logger=logging.getLogger()

@log
def hcp_load_data(proc_subj):
    
        if not hasattr(proc_subj, 'raw_rest'):
            
            filename = str(proc_subj.meg_rest_raw.fpath) + '/c,rf0.0Hz'
            proc_subj.raw_rest = mne.io.read_raw_bti(filename,
                                    head_shape_fname=None,convert=False,preload=True)
            for i in range(len(proc_subj.raw_rest.info['chs'])):                      # was processed similar to the HCP data
                        proc_subj.raw_rest.info['chs'][i]['coord_frame'] = 1
            proc_subj.raw_rest.pick_types(meg=True, eeg=False)
            
        if (not hasattr(proc_subj, 'raw_eroom')) and (proc_subj.meg_er_raw != None):
            
            filename_er = str(proc_subj.meg_er_raw.fpath)+ '/c,rf0.0Hz'
            proc_subj.raw_eroom = mne.io.read_raw_bti(filename_er,
                                    head_shape_fname=None,convert=False,preload=True)
            for i in range(len(proc_subj.raw_eroom.info['chs'])):                      # was processed similar to the HCP data
                        proc_subj.raw_eroom.info['chs'][i]['coord_frame'] = 1
            proc_subj.raw_eroom.pick_types(meg=True, eeg=False)

        # For subsequent reference, if raw_room not provided, set to None
        if (not hasattr(proc_subj, 'raw_eroom')):
            proc_subj.raw_eroom=None
        # figure out the MEG system vendor, note that this may be different from 
        # datatype if the datatype is .fif
        proc_subj.vendor = mne.channels.channels._get_meg_system(proc_subj.raw_rest.info)

@log
def proc_hcp_mri(proc_subj, t1_override=None,redo_all=False):
        
        trans_bids_path = proc_subj.bids_path.copy().update(datatype='anat',
                                                    session=proc_subj.session,
                                                    suffix=None,
                                                    extension=None)
        trans_bids_basename = proc_subj.bids_path.copy().update(datatype='anat',
                                                    session=None,
                                                    suffix=None,
                                                    extension=None)
        trans_path = str(trans_bids_path.directory) + '/' + trans_bids_basename.basename + '-head_mri-trans.fif'
                                                    
        # Configure filepaths
        os.environ['SUBJECTS_DIR']=proc_subj.subjects_dir
        deriv_path = proc_subj.rest_derivpath.copy()
        
        # do all the remaining freesurfer processing steps        
        for proc in proc_subj.anat_vars.process_list:
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
            proc_subj.rest_fwd = fwd
        
        watershed_check_path=op.join(proc_subj.subjects_dir,             # watershed bem
                                     f'sub-{proc_subj.subject}',
                                     'bem',
                                     'inner_skull.surf'
                                     )
        if (not os.path.exists(watershed_check_path)) or (redo_all is True):
            mne.bem.make_watershed_bem(f'sub-{proc_subj.subject}',
                                       proc_subj.subjects_dir,
                                       overwrite=True,
                                       gcaatlas=True
                                       )
        if (not bem_fname.fpath.exists()) or (redo_all is True):
            bem = mne.make_bem_model(f'sub-{proc_subj.subject}', 
                                     subjects_dir=proc_subj.subjects_dir, 
                                     conductivity=[0.3])
            bem_sol = mne.make_bem_solution(bem)
            
            mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
        else:
            bem_sol = mne.read_bem_solution(bem_fname)
            
        if (not src_fname.fpath.exists()) or (redo_all is True):    # source space
            src = mne.setup_source_space(f'sub-{proc_subj.subject}',
                                         spacing='oct6', add_dist='patch',
                                 subjects_dir=proc_subj.subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
        
        trans = mne.read_trans(trans_path)
        mne.write_trans(trans_fname.fpath, trans, overwrite=True)

        # make the forward solution
        fwd = mne.make_forward_solution(proc_subj.raw_rest.info, trans, src, bem_sol, eeg=False, 
                                        n_jobs=proc_subj._n_jobs)
        proc_subj.rest_fwd=fwd
        mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)

#%%  Argparse
if __name__=='__main__':
    import argparse  
    parser = argparse.ArgumentParser()
    standardargs = parser.add_argument_group('Standard Inputs')
    standardargs.add_argument('-bids_root',
                        help='''Top level directory of the bids data'''
                        )

    standardargs.add_argument('-subject',
                        help='''BIDS ID of subject to process''',
                        default=None
                        )
    standardargs.add_argument('-subjects_dir',
                        help='''Freesurfer subjects directory, only specify if not \
                        bids_root/derivatives/freesurfer/subjects'''
                        )
    standardargs.add_argument('-fs_subject',
                        help='''Freefurfer subject ID if different from BIDS ID'''
                        )
    standardargs.add_argument('-run',
                        help='''Run number.  \nNOTE: 01 is different from 1''',
                        default='1'
                        )
    standardargs.add_argument('-session',
                        default=None,
                        )
    standardargs.add_argument('-mains',
                        help='Electric mains frequency  (50 or 60)',
                        default=60.0,
                        type=float
                        )
    standardargs.add_argument('-rest_tag',
                        help='Override in case task name is other than rest\
                            for example - resteyesopen',
                        default='rest'
                        )
    standardargs.add_argument('-emptyroom_tag',
                        help='Override in case emptryoom is other than \
                            emptyroom.  In case of no emptyroom, set as None on cmdline',
                        default='emptyroom'
                        )
    standardargs.add_argument('-n_jobs',
                        help='''number of jobs to run concurrently for 
                        multithreaded operations''',
                        default=1
                        )
    csvargs = parser.add_argument_group('Inputs from a CSV')
    csvargs.add_argument('-proc_fromcsv',
                        help='''Loop over all subjects in the bids_root
                        and process. Requires CSV file with processing manifest''',
                        default=None
                        )
    
    altargs = parser.add_argument_group('Alternative Inputs')
    altargs.add_argument('-fs_ave_fids',
                        help='''If no fiducials have been localized to the mri
                        manually, this provides a coarse fit from the ave brain
                        which is fine tuned with the headshape.  This is less
                        acurate than a manually assessed fid placement''',
                        action='store_true',
                        default=False
                        )
    altargs.add_argument('-do_dics',
                        help='''If flag is present, do a DICS beamformer. Otherwise, do lcmv. ''',
                        action='store_true',
                        default=0
                        )
    altargs.add_argument('-ct_sparse', 
                         help='''(Elekta/MEGIN datasets) This is the ct_sparse.fif
                         file associated with the MEG system.  This is an override if
                         the file cannot be found in the sub/ses/meg BIDS directory or at the 
                         top level of the bids tree.''', 
                         default=None
                         )
    altargs.add_argument('-sss_cal',
                         help='''(Elekta/MEGIN datasets) This is the sss_cal.dat
                         file associated with the MEG system.  This is an override if
                         the file cannot be found in the sub/ses/meg BIDS directory or at the 
                         top level of the bids tree.''', 
                         default=None
                         )
    altargs.add_argument('-emptyroom_run',
                        help='Override in case run designation differs for \
                             the emptyroom and rest datasets',
                        default=None)
    qaargs = parser.add_argument_group('QA Inputs')
    qaargs.add_argument('-ica_manual_qa_prep',
                        help='''if flag is present, stop after ICA for manual QA''',
                        action='store_true',
                        default=0
                        )
    qaargs.add_argument('-process_manual_ica_qa',
                        help='''If flag is present, pick up analysis after performing manual ICA QA''',
                        action='store_true',
                        default=0
                        )

    parser.add_argument('-remove_old',
                        help='''If flag is present, remove any files from a prior run (excepting freesurfer data). ''',
                        action='store_true',
                        default=0
                        )
                                   
    args = parser.parse_args()
    
    logger=logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    
    n_jobs = args.n_jobs  #extract this from the configuration file
    os.environ['n_jobs'] = str(n_jobs)
    
    os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false' # necessary for headless operation
    
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
    
    args.subject=args.subject.replace('sub-','') # strip the sub- off for uniformity
    print(args.subject)
        
    if args.emptyroom_run == None:
        args.emptyroom_run = args.run
        
    if args.remove_old:
        print('Removing files from prior runs')
        logfilename = args.subject + '_ses-' + str(args.session) + '_log.txt'
        subprocess.call(['rm', os.path.join(log_dir, logfilename)])
        subject_enigmadir = 'sub-' + args.subject
        enigmadir = os.path.join(bids_root,'derivatives/ENIGMA_MEG')
        subprocess.call(['rm','-r', os.path.join(enigmadir, subject_enigmadir)])
                        
    print('processing a single subject %s' % args.subject)      
                  
    logger = get_subj_logger(args.subject, args.session,args.rest_tag, args.run, log_dir)
    logger.info(f'processing subject {args.subject} session {args.session}')
    logger.info('Initializing structure')
    
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
            fs_ave_fids=args.fs_ave_fids,
            do_dics=args.do_dics)
    
    hcp_load_data(proc)
    
    proc.vendor_prep()
    proc.do_ica()
    proc.do_classify_ica()
    proc.do_preproc()
    proc.do_clean_ica()
    proc.do_proc_epochs()
    
    proc_hcp_mri(proc)
    
    proc.do_beamformer()
    proc.do_make_aparc_sub()
    proc.do_label_psds()
    proc.do_spectral_parameterization()
    proc.do_mri_segstats()
    proc.cleanup()
        



