#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:36:28 2023

@author: Allison Nugent
"""

import mne
import os.path as op
from mne_bids import BIDSPath
from mne.viz import Brain
import matplotlib.pyplot as plt
from enigmeg.process_meg import load_data

def gen_coreg_pngs(subjstruct):
    
    from mne.viz._brain.view import views_dicts
    from mne.viz import set_3d_view
    
    subjid = subjstruct.subject
    
    subjstruct.raw_rest = load_data(subjstruct.meg_rest_raw.fpath)
   
    subjstruct.trans = mne.read_trans(subjstruct.fnames['rest_trans'])
   
    fig = mne.viz.plot_alignment(info=subjstruct.raw_rest.info, trans=subjstruct.trans, subject='sub-'+subjid, 
                                 subjects_dir=subjstruct.subjects_dir)
    set_3d_view(fig,**views_dicts['both']['frontal'])
    img1=fig.plotter.screenshot()
    fig.plotter.close()
    fig = mne.viz.plot_alignment(info=subjstruct.raw_rest.info, trans=subjstruct.trans, subject='sub-'+subjid, 
                                 subjects_dir=subjstruct.subjects_dir)
    set_3d_view(fig,**views_dicts['both']['lateral'])
    img2=fig.plotter.screenshot()
    fig.plotter.close()
    fig = mne.viz.plot_alignment(info=subjstruct.raw_rest.info, trans=subjstruct.trans, subject='sub-'+subjid, 
               subjects_dir=subjstruct.subjects_dir)
   
    set_3d_view(fig,**views_dicts['both']['medial'])
    img3=fig.plotter.screenshot()
    fig.plotter.close()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img1)
    tmp=ax[0].axis('off')
    ax[1].imshow(img2)
    tmp=ax[1].axis('off')
    ax[2].imshow(img3)
    tmp=ax[2].axis('off')
   
    png_path=op.join(subjstruct.deriv_root,'ENIGMA_MEG_QA/' + 'sub-'+subjid)
    figname = op.join(png_path, 'sub-' + subjid + '_coreg.png')
    
    fig.savefig(figname, dpi=300,bbox_inches='tight')
    plt.close(fig)
   
def gen_bem_pngs(subjstruct):
    
    from mne.viz import plot_bem

    
    subjid = subjstruct.subject
    
    enigma_root = op.join(subjstruct.deriv_root, 'ENIGMA_MEG')
    enigma_subj_path = op.join(enigma_root, 'sub-' + subjid)
    src_path = op.join(enigma_subj_path, 'meg/sub-' + subjid + '_src.fif')
    src = mne.read_source_spaces(src_path, subjstruct.subjects_dir)
        
    png_path=op.join(subjstruct.deriv_root,'ENIGMA_MEG_QA/' + 'sub-'+subjid)
    figname_bem = op.join(png_path, 'sub-' + subjid + '_bem.png')
    
    fig=plot_bem(subject='sub-'+subjid, subjects_dir=subjstruct.subjects_dir, brain_surfaces='white', 
             slices=[50, 100, 150, 200], show=False, show_indices=True, mri='T1.mgz', show_orientation=True)
    fig.savefig(figname_bem)

def gen_src_pngs(subjstruct):
    
    from mne.viz._brain.view import views_dicts
    from mne.viz import set_3d_view
    from matplotlib.gridspec import GridSpec
    import matplotlib.image as img
    
    subjid = subjstruct.subject
    
    enigma_root = op.join(subjstruct.deriv_root, 'ENIGMA_MEG')
    enigma_subj_path = op.join(enigma_root, 'sub-' + subjid)
    src_path = op.join(enigma_subj_path, 'meg/sub-' + subjid + '_src.fif')
    src = mne.read_source_spaces(src_path, subjstruct.subjects_dir)

    png_path=op.join(subjstruct.deriv_root,'ENIGMA_MEG_QA/' + 'sub-'+subjid)
    figname_src = op.join(png_path, 'sub-' + subjid + '_src.png')

    fig=src.plot(subjects_dir=subjstruct.subjects_dir)
    set_3d_view(fig,**views_dicts['both']['frontal'])
    img1=fig.plotter.screenshot()
    fig.plotter.close()
    fig=src.plot(subjects_dir=subjstruct.subjects_dir)
    set_3d_view(fig,**views_dicts['both']['lateral'])
    img2=fig.plotter.screenshot()
    fig.plotter.close()
    fig=src.plot(subjects_dir=subjstruct.subjects_dir)
    set_3d_view(fig,**views_dicts['both']['medial'])
    img3=fig.plotter.screenshot()
    fig.plotter.close()
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img1)
    tmp=ax[0].axis('off')
    ax[1].imshow(img2)
    tmp=ax[1].axis('off')
    ax[2].imshow(img3)
    tmp=ax[2].axis('off')
    fig.savefig(figname_src, dpi=300,bbox_inches='tight')
    plt.close(fig)

def gen_surf_pngs(subjstruct):
    
    Brain = mne.viz.get_brain_class()
    
    subjid = subjstruct.subject
    
    png_path=op.join(subjstruct.deriv_root,'ENIGMA_MEG_QA/' + 'sub-'+subjid)
    figname_surf = op.join(png_path, 'sub-' + subjid + '_surf.png')
    
    labels = mne.read_labels_from_annot('sub-'+subjid, subjects_dir=subjstruct.subjects_dir,
                                        parc='aparc', hemi='both',surf_name='white')
    
    
    brain = Brain('sub-'+subjid, 'lh','pial',subjects_dir=subjstruct.subjects_dir,cortex='classic',
                  background='white', views='lateral')
    img1=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'lh','pial',subjects_dir=subjstruct.subjects_dir,cortex='classic',
                  background='white', views='medial')
    img2=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'rh','pial',subjects_dir=subjstruct.subjects_dir,cortex='classic',
                  background='white', views='lateral')
    img3=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'rh','pial',subjects_dir=subjstruct.subjects_dir,cortex='classic',
                  background='white', views='medial')
    img4=brain.screenshot()
    brain.close()
    
    
    brain = Brain('sub-'+subjid, 'lh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='lateral')
    brain.add_annotation('aparc_sub')
    img5=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'lh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='medial')
    brain.add_annotation('aparc_sub')
    img6=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'rh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='lateral')
    brain.add_annotation('aparc_sub')
    img7=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'rh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='medial')
    brain.add_annotation('aparc_sub')
    img8=brain.screenshot()
    brain.close()
    
    fig, ax = plt.subplots(2,4)
    ax[0][0].imshow(img1)
    tmp=ax[0][0].axis('off')
    ax[0][1].imshow(img2)
    tmp=ax[0][1].axis('off')
    ax[0][2].imshow(img3)
    tmp=ax[0][2].axis('off')
    ax[0][3].imshow(img4)
    tmp=ax[0][3].axis('off')
    ax[1][0].imshow(img5)
    tmp=ax[1][0].axis('off')
    ax[1][1].imshow(img6)
    tmp=ax[1][1].axis('off')  
    ax[1][2].imshow(img7)
    tmp=ax[1][2].axis('off')
    ax[1][3].imshow(img8)
    tmp=ax[1][3].axis('off')
    plt.tight_layout()
    fig.savefig(figname_surf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def gen_epo_pngs(subjstruct):
    
    subjid = subjstruct.subject
    
    png_path=op.join(subjstruct.deriv_root,'ENIGMA_MEG_QA/' + 'sub-'+subjid)
    figname_epo_psd = op.join(png_path, 'sub-' + subjid + '_spectra.png')
    
    epo_path = subjstruct.rest_derivpath.copy().update(suffix='epo', extension='.fif')
    epochs = mne.read_epochs(epo_path)
    
    fig = epochs.compute_psd(fmin=subjstruct.proc_vars['fmin'],fmax=subjstruct.proc_vars['fmax']).plot(picks='meg')
    fig.savefig(figname_epo_psd, dpi=300, bbox_inches='tight')

def gen_fooof_pngs(subjstruct):
    
    import pandas as pd
    import nibabel as nib
    import numpy as np
    
    subjid = subjstruct.subject
    
    fooof_dir = subjstruct.deriv_root + '/ENIGMA_MEG/sub-' +subjid + '/meg/fooof_results'
    fooof_results = op.join(fooof_dir, 'Band_rel_power.csv')
    fooof_dframe = pd.read_csv(fooof_results, delimiter='\t')  
    fooof_dframe = fooof_dframe.rename(columns={'Unnamed: 0':'Parcel'})
    
    fooof_dframe_lh = fooof_dframe[fooof_dframe['Parcel'].str.contains('-lh')].reset_index(drop=True)
    fooof_dframe_rh = fooof_dframe[fooof_dframe['Parcel'].str.contains('-rh')].reset_index(drop=True)
    
    aparc_file_lh = op.join(subjstruct.subjects_dir, 'sub-'+subjid, 'label', 'lh.aparc_sub.annot')
    aparc_file_rh = op.join(subjstruct.subjects_dir, 'sub-'+subjid, 'label', 'rh.aparc_sub.annot')
    
    labels_lh, ctab, names_lh = nib.freesurfer.read_annot(aparc_file_lh)
    names2_lh=[i.decode() for i in names_lh] #convert from binary
    if 'corpuscallosum' in names2_lh: names2_lh.remove('corpuscallosum')
    if 'unknown' in names2_lh: names2_lh.remove('unknown')
    labels_rh, ctab, names_rh = nib.freesurfer.read_annot(aparc_file_rh)
    names2_rh=[i.decode() for i in names_rh] #convert from binary
    if 'corpuscallosum' in names2_rh: names2_rh.remove('corpuscallosum')
    if 'unknown' in names2_rh: names2_rh.remove('unknown')
    
    roi_data_lh = np.zeros(600)
    roi_data_rh = np.zeros(600)
    
    for idx,name in enumerate(names2_lh):
        if fooof_dframe_lh.loc[idx, 'Parcel'].split('-')[0] != name:
            print(f'{fooof_dframe.loc[idx, "Parcel"].split("-")[0]} != {name}')
            raise(ValueError)
        roi_data_lh[idx]=fooof_dframe_lh.loc[idx, '[8, 12]']
    for idx,name in enumerate(names2_rh):
        if fooof_dframe_rh.loc[idx, 'Parcel'].split('-')[0] != name:
            print(f'{fooof_dframe.loc[idx, "Parcel"].split("-")[0]} != {name} at idx %d' % idx)
            raise(ValueError)
        roi_data_rh[idx]=fooof_dframe_rh.loc[idx, '[8, 12]']   
        
    vtx_data_lh = roi_data_lh[labels_lh]
    vtx_data_lh[labels_lh == -1] = 0
    thresh = 0.001
    vtx_data_lh[np.abs(vtx_data_lh)<thresh] = 0
    vtx_min_lh = np.min(vtx_data_lh[np.nonzero(vtx_data_lh)])   
    vtx_data_rh = roi_data_rh[labels_rh]
    vtx_data_rh[labels_rh == -1] = 0
    thresh = 0.001
    vtx_data_rh[np.abs(vtx_data_rh)<thresh] = 0
    vtx_min_rh = np.min(vtx_data_rh[np.nonzero(vtx_data_rh)])   
    
    brain = Brain('sub-'+subjid, 'lh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='lateral')
    brain.add_data(vtx_data_lh, colormap='coolwarm', alpha=1, fmin=vtx_min_lh)
    img1=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'lh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='medial')
    brain.add_data(vtx_data_lh, colormap='coolwarm', alpha=1, fmin=vtx_min_lh)
    img2=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'rh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='lateral')
    brain.add_data(vtx_data_rh, colormap='coolwarm', alpha=1, fmin=vtx_min_rh)
    img3=brain.screenshot()
    brain.close()
    brain = Brain('sub-'+subjid, 'rh','inflated',subjects_dir=subjstruct.subjects_dir,cortex='low_contrast',
                  background='white', views='medial')
    brain.add_data(vtx_data_rh, colormap='coolwarm', alpha=1, fmin=vtx_min_rh)
    img4=brain.screenshot()
    brain.close()
    
    png_path=op.join(subjstruct.deriv_root,'ENIGMA_MEG_QA/' + 'sub-'+subjid)
    figname_alpha = op.join(png_path, 'sub-' + subjid + '_beamformer.png')
    
    fig, ax = plt.subplots(2,2)
    ax[0][0].imshow(img1)
    tmp=ax[0][0].axis('off')
    ax[0][1].imshow(img2)
    tmp=ax[0][1].axis('off')
    ax[1][0].imshow(img3)
    tmp=ax[1][0].axis('off')
    ax[1][1].imshow(img4)
    tmp=ax[1][1].axis('off')
    plt.tight_layout()
    #plt.show()
    fig.savefig(figname_alpha, dpi=300,bbox_inches='tight')
    plt.close(fig)