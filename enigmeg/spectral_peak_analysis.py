#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:30:08 2020

@author: stoutjd
"""

import mne
from scipy.signal import welch

from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model



def calc_spec_peak(freqs, powers, fitting_bw=[1,55], out_image_path=None):
    '''Spectral fitting routine from the FOOOF toolbox
    https://fooof-tools.github.io/fooof/index.html
    
        
    doi: https://doi.org/10.1101/299859 
    
    Fit the spectral peaks and return the fit parameters
    Save the image of the spectral data and peak fits
    
    Inputs:
        freqs - numpy array of frequencies
        powers - numpy array of spectral power
        fitting_bw - Reduce the total bandwidth to fit the 1/f and spectral peaks
        
    Outputs:
        params - parameters from the FOOOF fit
        
    '''
    
    #Crop Frequencies for 1/f
    powers=powers[(freqs>fitting_bw[0]) & (freqs<=fitting_bw[1])]
    freqs=freqs[(freqs>fitting_bw[0]) & (freqs<=fitting_bw[1])]
    
    # Initialize power spectrum model objects and fit the power spectra
    fm1 = FOOOF(min_peak_height=0.05, verbose=False)
    fm1.fit(freqs, powers)

    if out_image_path is not None: 
        import matplotlib
        from matplotlib import pylab
        matplotlib.use('Agg')
        # import pylab
        fig = pylab.Figure(figsize=[10,6]) #, dpi=150)
        ax = fig.add_subplot()
        plot_annotated_model(fm1, annotate_peaks=False, ax=ax)
        fig.tight_layout()
        fig.savefig(out_image_path, dpi=150, bbox_inches="tight")
        
    
    params=fm1.get_results()

    # plot_spectrum(freqs, powers, log_powers=True,
    #               color='black', label='Original Spectrum')
    return params


def test_calc_spec_peak():
    line_freq=60
    raw = mne.io.read_raw_ctf('/data/MEG/20200122/APBWVFAR_rest_20200122_03.ds', 
                              preload=True)
    raw.pick_channels(['MRP44-1609'])
    raw.resample(300)
    raw.filter(1.0, None)
    raw.notch_filter(line_freq)
    freqs, powers =  welch(raw._data, fs=raw.info['sfreq'], window='hanning')
    powers=powers.squeeze()    
    tmp_param = calc_spec_peak(freqs, powers)
    
    assert round(tmp_param.peak_params[0][0], ndigits=1) == 9.4
    




