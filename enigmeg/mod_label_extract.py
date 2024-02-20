#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:26:19 2021

@author: MNE python and A Nugent
"""
import copy
import numpy as np

from mne.fixes import _safe_svd
from mne.source_space import SourceSpaces
from mne.utils import (logger, _validate_type)
from types import GeneratorType
import scipy

#from mne import source_estimate as mod_source_estimate

from mne.source_estimate import (_pca_flip, _volume_labels,
        _BaseSourceEstimate,_BaseVolSourceEstimate,_BaseVectorSourceEstimate,
        _prepare_label_extraction)

from numpy import linalg
from mne.time_frequency import psd_array_multitaper
from scipy.stats import trim_mean
import os

num_freq_bins=177  #Hardcoded freq bins - Bad form

def _pca15_fft(flip, data, fmin, fmax, srate):
    U, s, V = linalg.svd(data, full_matrices=False)
    
    maxeig=15
    if 'n_jobs' in os.environ:
        n_jobs = int(os.environ['n_jobs'])
    else:
        n_jobs=1
    # use average power in label for scaling
    epoch_spectra, freq_bins = psd_array_multitaper(V[0:maxeig], 
                                                srate,                    #!!!!################ HardCodede
                                                fmin=fmin, fmax=fmax,
                                                bandwidth=2, 
                                                n_jobs=n_jobs, 
                                                adaptive=True, 
                                                low_bias=True, 
                                                normalization='full',
                                                verbose='warning') 
    
    eigval_weighted_spectra=s[0:maxeig,np.newaxis]*epoch_spectra
    
    # Reject top and bottom 10% using trimmed mean
    output_spectra = trim_mean(eigval_weighted_spectra, 0.1, axis=0)
    # Normalize by number of samples
    normalized_spectra = output_spectra / np.sqrt(len(data))
    return  output_spectra 

from mne.source_estimate import _label_funcs
_label_funcs['pca15_multitaper']=_pca15_fft
#_label_funcs=mod_source_estimate._label_funcs

'''
The code below is from mne python 0.21.1
Changes were made to utilize the extract label function while
modifying the extraction process.  

This will be monkeypatched into the processing before calling

'''

def mod_gen_extract_label_time_course(stcs, labels, src, *, mode='mean',
                                   allow_empty=False, trans=None,
                                   mri_resolution=True, verbose=None,
                                   fmin=1, fmax=45):
    # loop through source estimates and extract time series
    #from scipy import sparse
    #if src is None and mode in ['mean', 'max', 'pca']:
    #    kind = 'surface'
    #else:
    #    _validate_type(src, SourceSpaces)
    #    kind = src.kind
    #_check_option('mode', mode, _get_default_label_modes())

    #if kind in ('surface', 'mixed'):
    #    if not isinstance(labels, list):
    #        labels = [labels]
    #    use_sparse = False
    #else:
    #    labels = _volume_labels(src, labels, mri_resolution)
    #    use_sparse = bool(mri_resolution)
    n_mode = len(labels)  # how many processed with the given mode
    n_mean = 0 # len(src[2:]) if kind == 'mixed' else 0
    n_labels = n_mode + n_mean
    vertno = func = None
    for si, stc in enumerate(stcs):
        #_validate_type(stc, _BaseSourceEstimate, 'stcs[%d]' % (si,),
        #               'source estimate')
        #_check_option(
        #    'mode', mode, _get_allowed_label_modes(stc),
        #    'when using a vector and/or volume source estimate')
        #if isinstance(stc, (_BaseVolSourceEstimate,
        #                    _BaseVectorSourceEstimate)):
        #    mode = 'mean' if mode == 'auto' else mode
        #else:
        #    mode = 'mean_flip' if mode == 'auto' else mode
        n_timepoints = np.shape(stc.data)[1]
        freqs = scipy.fft.rfftfreq(n_timepoints, stc.tstep)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        n_bins=len(freqs)
        srate = 1/stc.tstep
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)  # avoid keeping a ref
            nvert = np.array([len(v) for v in vertno])
            label_vertidx, src_flip = _prepare_label_extraction(
                stc, labels, src, mode, allow_empty, False)
            func = _label_funcs[mode]
        # make sure the stc is compatible with the source space
        if len(vertno) != len(stc.vertices):
            raise ValueError('stc not compatible with source space')
        for vn, svn in zip(vertno, stc.vertices):
            if len(vn) != len(svn):
                raise ValueError('stc not compatible with source space. '
                                 'stc has %s time series but there are %s '
                                 'vertices in source space. Ensure you used '
                                 'src from the forward or inverse operator, '
                                 'as forward computation can exclude vertices.'
                                 % (len(svn), len(vn)))
            if not np.array_equal(svn, vn):
                raise ValueError('stc not compatible with source space')

        #logger.info('Extracting time courses for %d labels (mode: %s)'
        #            % (n_labels, mode))

        # do the extraction
        label_tc = np.zeros((n_labels,) + (n_bins,),
                            dtype=stc.data.dtype)
        for i, (vertidx, flip) in enumerate(zip(label_vertidx, src_flip)):
            if vertidx is not None:
                #if isinstance(vertidx, sparse.csr_matrix):
                #    assert mri_resolution
                #    assert vertidx.shape[1] == stc.data.shape[0]
                #    this_data = np.reshape(stc.data, (stc.data.shape[0], -1))
                #    this_data = vertidx @ this_data
                #    this_data.shape = \
                #        (this_data.shape[0],) + stc.data.shape[1:]
                #else:
                this_data = stc.data[vertidx]
                label_tc[i] = func(flip, this_data, fmin, fmax, srate)

        # extract label time series for the vol src space (only mean supported)
        offset = nvert[:-n_mean].sum()  # effectively :2 or :0
        for i, nv in enumerate(nvert[2:]):
            if nv != 0:
                v2 = offset + nv
                label_tc[n_mode + i] = np.mean(stc.data[offset:v2], axis=0)
                offset = v2

        # this is a generator!
        yield label_tc

def mod_extract_label_time_course(
    stcs,
    labels,
    src,
    mode="auto",
    allow_empty=False,
    return_generator=False,
    *,
    mri_resolution=True,
    verbose=None,
    fmin=1,
    fmax=45
):
    """Extract label time course for lists of labels and source estimates.

    This function will extract one time course for each label and source
    estimate. The way the time courses are extracted depends on the mode
    parameter (see Notes).

    Parameters
    ----------
    stcs : SourceEstimate | list (or generator) of SourceEstimate
        The source estimates from which to extract the time course.
    %(labels_eltc)s
    %(src_eltc)s
    %(mode_eltc)s
    %(allow_empty_eltc)s
    return_generator : bool
        If True, a generator instead of a list is returned.
    %(mri_resolution_eltc)s
    %(verbose)s

    Returns
    -------
    %(label_tc_el_returns)s

    Notes
    -----
    %(eltc_mode_notes)s

    If encountering a ``ValueError`` due to mismatch between number of
    source points in the subject source space and computed ``stc`` object set
    ``src`` argument to ``fwd['src']`` or ``inv['src']`` to ensure the source
    space is the one actually used by the inverse to compute the source
    time courses.
    """
    # convert inputs to lists
    if not isinstance(stcs, (list, tuple, GeneratorType)):
        stcs = [stcs]
        return_several = False
        return_generator = False
        print('not a generator')
    else:
        print('return several')
        return_several = True
    print('running mod_gen_extract_label_time_course')
    label_tc = mod_gen_extract_label_time_course(
        stcs,
        labels,
        src,
        mode=mode,
        allow_empty=allow_empty,
        mri_resolution=mri_resolution,
        fmin=fmin,
        fmax=fmax
    )

    if not return_generator:
        # do the extraction and return a list
        label_tc = list(label_tc)

    if not return_several:
        # input was a single SoureEstimate, return single array
        label_tc = label_tc[0]

    return label_tc


#mod_source_estimate._gen_extract_label_time_course=mod_gen_extract_label_time_course
