#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:08:12 2020

@author: stoutjd
"""
import os
import os.path as op
import glob
import mne

class anat_info():
    '''Collect information for processing the data'''
    def __init__(self, **kwargs):
        self.recon1=False
        self.recon2=False
        self.recon3=False
        self.setup_source=False
        self.run_unprocessed=False
        self.subjid=kwargs['subjid']
        if 'SUBJECTS_DIR' in kwargs:
            self.subjects_dir=kwargs['SUBJECTS_DIR']
        else:
            self.subjects_dir=os.environ['SUBJECTS_DIR']
        if self.subjid not in os.listdir(self.subjects_dir):
            raise ValueError('''{} not in {}.  If unexpected: 
                1) check that the subject is in the SUBJECTS_DIR, or 
                2) Set subjects_dir at the commandline'''.format(self.subjid, self.subjects_dir))
        self.fs_subj_dir=os.path.join(self.subjects_dir, self.subjid)
        self.fs_mri_contents=os.listdir(os.path.join(self.fs_subj_dir, 'mri')) 
        self.fs_surf_contents=os.listdir(os.path.join(self.fs_subj_dir, 'surf'))
        self.fs_label_contents=os.listdir(os.path.join(self.fs_subj_dir, 'label'))
        
        self.outfolder = os.path.join(os.environ['ENIGMA_REST_DIR'], self.subjid)
        self.pickle_file = os.path.join(self.outfolder, 'info.pkl')
        #Setup output expectations
        # self.recon1_outputs
        # self.recon2_outputs
        # self.recon3_outputs
        self.fs_bem_dir=os.path.join(self.fs_subj_dir, 'bem')
        self.run_make_watershed_bem = not os.path.exists(os.path.join(self.fs_bem_dir,
                                                                'inner_skull.surf'))
        self.src_filename = os.path.join(self.outfolder, 'source_space-src.fif')
        self.run_make_src = not os.path.exists(self.src_filename)
        self.trans = None
        
        self.bem_sol_filename = os.path.join(self.outfolder, 'bem_sol-sol.fif')
        self.run_bem_sol = not os.path.exists(self.bem_sol_filename)

def compile_fs_process_list(info):
    '''Verifies necessary steps for processing and returns a list'''
    process_steps=[]
    proc_downstream=0
    if info.run_unprocessed:
        if ('brainmask.mgz' not in info.fs_mri_contents) | info.recon1:
            process_steps.append('recon-all -autorecon1 -s {}'.format(info.subjid))
            proc_downstream = True
        if ('lh.pial' not in info.fs_surf_contents) | ('rh.pial' not in info.fs_surf_contents) | info.recon2 | proc_downstream:
            process_steps.append('recon-all -autorecon2 -s {}'.format(info.subjid))
            proc_downstream = True
        if ('lh.aparc.annot' not in info.fs_label_contents) | ('rh.aparc.annot' not in info.fs_label_contents) | info.recon3 | proc_downstream:
            process_steps.append('recon-all -autorecon3 -s {}'.format(info.subjid))
            proc_downstream = True
            
    # If run_unprocessed is not set.  All independent steps must best to run manually
    if info.recon1:
        process_steps.append('recon-all -autorecon1 -s {}'.format(info.subjid))
    if info.recon2:
        process_steps.append('recon-all -autorecon2 -s {}'.format(info.subjid))
    if info.recon3:
        process_steps.append('recon-all -autorecon3 -s {}'.format(info.subjid))
    return process_steps     


def write_aparc_sub(subjid=None, subjects_dir=None):
    '''Check for fsaverage and aparc_sub and download
    Morph fsaverage aparc_sub labels to single subject data
    
    https://mne.tools/stable/auto_examples/visualization/plot_parcellation.html
    '''
    mne.datasets.fetch_fsaverage(verbose='ERROR') #True requires TQDM
    mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose='ERROR')
    
    sub_labels=mne.read_labels_from_annot('fsaverage',parc='aparc_sub', 
                                   subjects_dir=subjects_dir)        
    subject_labels=mne.morph_labels(sub_labels, subject_to=subjid, 
                                 subjects_dir=subjects_dir)
    mne.write_labels_to_annot(subject_labels, subject=subjid, 
                              parc='aparc_sub', subjects_dir=subjects_dir, 
                              overwrite=True)

def subcommand(function_str):
    from subprocess import check_call
    check_call(function_str.split(' '))

def pickle_info(info):
    import pickle, datetime
    now=datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    if os.path.exists(info.pickle_file):
        os.rename(info.pickle_file, info.pickle_file+'BAK'+now)
    fid=open(info.pickle_file, 'wb')
    pickle.dump(info, fid)
    fid.close()
    
if __name__=='__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjects_dir', help='''Freesurfer subjects_dir can be 
                        assigned at the commandline if not already exported.''')
    parser.add_argument('-subjid', help='''Define subjects id (folder name)
                        in the SUBJECTS_DIR''')
    parser.add_argument('-recon_check', help='''Process all anatomical steps that
                        have not been completed already.  This will check the major
                        outputs from autorecon1, 2, 3, and mne source setup and
                        proceed with the processing. The default is set to TRUE''')
    parser.add_argument('-recon1', help='''Force recon1 to be processed''', action='store_true')
    parser.add_argument('-recon2', help='''Force recon2 to be processed''', action='store_true')
    parser.add_argument('-recon3', help='''Force recon3 to be processed''', action='store_true')
    parser.add_argument('-setup_source', help='''Runs the setup source space processing
                        in mne python to create the BEM model''', action='store_true')
    parser.add_argument('-run_unprocessed', help='''Checks for all unrun processes and
                        runs any additional steps for inputs to the source model''', action='store_true')
    parser.description='''Processing for the anatomical inputs of the enigma pipeline'''
    args = parser.parse_args()
    if not args.subjid: raise ValueError('Subject ID must be set')
    if not args.subjects_dir: 
        args.subjects_dir=os.environ['SUBJECTS_DIR']
    else:
        os.environ['SUBJECTS_DIR']=args.subjects_dir  
    
    #Initialize Defaults
    info=anat_info(subjid=args.subjid, SUBJECTS_DIR=args.subjects_dir)
    #Override defaults with commandline options
    if args.recon1: info.recon1=True
    if args.recon2: info.recon2=True
    if args.recon3: info.recon3=True
    if args.setup_source: info.setup_source=True
    if args.run_unprocessed: info.run_unprocessed=True
    
    ## Create the subprocess loops and run the freesurfer commands
    fs_proc_list=compile_fs_process_list(info)
    if 'IsRunning.lh+rh' in os.listdir(os.path.join(info.fs_subj_dir, 'scripts')):
        del_run_file=input('''The IsRunning.lh+rh file is present.  Could be from a broken process \
              or the process is currently running.  Do you want to delete to continue?(y/n)''')
        if del_run_file.lower()=='y':
            os.remove(os.path.join(info.fs_subj_dir, 'scripts','IsRunning.lh+rh'))
    for proc in fs_proc_list:
        subcommand(proc)
    
    # Run the BEM processing steps
    if info.run_make_watershed_bem: mne.bem.make_watershed_bem(info.subjid, subjects_dir=info.subjects_dir)
        
    # Create MEG related output folder
    if not os.path.exists(info.outfolder): os.mkdir(info.outfolder)
    
    # Run the source
    if not info.run_make_src:
        src = mne.source_space.read_source_spaces(info.src_filename)
    else:
        src = mne.setup_source_space(info.subjid, spacing='oct6', add_dist='patch',
                                 subjects_dir=info.subjects_dir)
        src.save(info.src_filename)
    
    # Run bem solution
    if not info.run_bem_sol:
        print('BEM already created: skipping')
        #bem = mne.read_bem_solution(info.bem_sol_filename)
    else:
        conductivity = (0.3,)
        model = mne.make_bem_model(subject=info.subjid, ico=4,
                                    conductivity=conductivity,
                                    subjects_dir=info.subjects_dir)
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(info.bem_sol_filename, bem)
        
    pickle_info(info) 

    # Run aparc_sub processing to create 450 parcels
    path_bool = os.path.join(info.subjects_dir, info.subjid, 'label', 
                       'lh.aparc_sub.annot')
    if not os.path.exists(path_bool):
        write_aparc_sub(subjid=info.subjid, subjects_dir=info.subjects_dir) 
        
        
    if not glob.glob(op.join(info.outfolder, '*.html')):
        report = mne.Report(subject=info.subjid, subjects_dir=info.subjects_dir, 
                            verbose=True)
        report.parse_folder(info.outfolder, pattern='', mri_decim=25)
        out_html_file=op.join(info.outfolder, 'report_mri_bem.html')
        report.save(out_html_file, overwrite=True)
        
    

        
        
        
        

    
    
        
    

        
    
        


    


