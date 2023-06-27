#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:50:36 2023

@author: nugenta
"""

import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os.path as op

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ica_fname',
                        help='''Top level directory of the bids data'''
                        )
    parser.add_argument('-raw_fname',
                        help='''BIDS ID of subject to process'''
                        )
    parser.add_argument('-vendor',
                        help='''Freesurfer subjects directory, only specify if not \
                        bids_root/derivatives/freesurfer/subjects'''
                        )
    parser.add_argument('-results_dir',
                        help='''Freefurfer subject ID if different from BIDS ID'''
                        )
    parser.add_argument('-basename',
                        help='''filename base'''
                        )
    args = parser.parse_args()
   
    
    ica_fname = args.ica_fname
    raw_fname = args.raw_fname
    vendor = args.vendor
    results_dir = args.results_dir
    basename = args.basename
    
    ica = mne.preprocessing.read_ica(ica_fname)
    raw = mne.io.read_raw(raw_fname)
    comps = ica.get_sources(raw).get_data()
    topos = ica.get_components()
    raw.pick_types(meg=True,exclude='bads')
        
    if(vendor=='306m'):
        picks = mne.pick_types(raw.info,meg='mag')
        picks_idx = mne.viz.ica._picks_to_idx(raw.info, picks)
        topos = topos[picks_idx,:]
        raw.pick_types(meg='mag')
        
    for compidx in range(20):
        
        fig = plt.figure(layout='constrained')
        gs = GridSpec(1,5,figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1:6])
        ax2.plot(comps[compidx,0:5000])
        mne.viz.plot_topomap(topos[:,compidx],raw.info, axes=ax1, show=False)
        
        pngfname = f'{basename}_icacomp-{compidx}.png'
        print(pngfname)
        path = op.join(results_dir,pngfname)
        
        fig.savefig(path, dpi=300, bbox_inches='tight')

