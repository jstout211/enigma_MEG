#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:26:19 2021

@author: stoutjd
"""
#!
######
# Need to determine the number of bins for freqs to use as output of extract labels
# Need to also determine the sampling frequency - currently hard coded in _pca15_fft
####

import mne
from mne import source_estimate as mod_source_estimate
import numpy as np
from numpy import linalg
from mne.time_frequency import psd_array_multitaper



num_freq_bins=177  #Hardcoded freq bins - Bad form



def _pca15_fft(flip, data):
    U, s, V = linalg.svd(data, full_matrices=False)
    
    maxeig=15
    
    # determine sign-flip
    #sign = np.sign(np.dot(U[:, 0], flip))
    # use average power in label for scaling
    epoch_spectra, freq_bins = psd_array_multitaper(V[0:maxeig], 
                                                300,                    #!!!!################ HardCodede
                                                fmin=1, fmax=45,
                                                bandwidth=2, 
                                                n_jobs=4, 
                                                adaptive=True, 
                                                low_bias=True) 
    
    # scale = linalg.norm(s) / np.sqrt(len(data))
    normalized_spectra=s[0:maxeig,np.newaxis]*epoch_spectra
    output_spectra = np.mean(normalized_spectra, axis=0)
    return  output_spectra #s[0:maxeig,np.newaxis]* V[0:maxeig] #sign *

from mne.source_estimate import _label_funcs
_label_funcs['pca15_multitaper']=_pca15_fft



'''
The code below is from mne python 0.21.1
Changes were made to utilize the extract label function while
modifying the extraction process.  

This will be monkeypatched into the processing before calling

'''
from mne.source_estimate import _validate_type, _check_option, _volume_labels,\
    SourceSpaces, _prepare_label_extraction
import copy
import numpy as np
_label_funcs=mod_source_estimate._label_funcs
    
def mod_gen_extract_label_time_course(stcs, labels, src, mode='mean',
                                   allow_empty=False, trans=None,
                                   mri_resolution=True, verbose=None):
    # loop through source estimates and extract time series
    # _validate_type(src, SourceSpaces)
    # _check_option('mode', mode, sorted(_label_funcs.keys()) + ['auto'])

    # kind = src.kind
    # if kind in ('surface', 'mixed'):
    #     if not isinstance(labels, list):
    #         labels = [labels]
    #     use_sparse = False
    # else:
    #     labels = _volume_labels(src, labels, trans, mri_resolution)
    #     use_sparse = bool(mri_resolution)
    n_mode = len(labels)  # how many processed with the given mode
    n_mean = 0 # len(src[2:]) #if kind == 'mixed' else 0
    n_labels = n_mode + n_mean
    vertno = func = None
    for si, stc in enumerate(stcs):
    #     _validate_type(stc, _BaseSourceEstimate, 'stcs[%d]' % (si,),
    #                    'source estimate')
    #     if isinstance(stc, (_BaseVolSourceEstimate,
    #                         _BaseVectorSourceEstimate)):
    #         _check_option(
    #             'mode', mode, ('mean', 'max', 'auto'),
    #             'when using a vector and/or volume source estimate')
    #         mode = 'mean' if mode == 'auto' else mode
    #     else:
    #         mode = 'mean_flip' if mode == 'auto' else mode
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

        # logger.info('Extracting time courses for %d labels (mode: %s)'
        #             % (n_labels, mode))

        # CHANGES >>
        # do the extraction
        label_tc = np.zeros((n_labels,) + (num_freq_bins,),
                            dtype=stc.data.dtype)        
        # label_tc = np.zeros((n_labels,) + stc.data.shape[1:],
        #                     dtype=stc.data.dtype)
        for i, (vertidx, flip) in enumerate(zip(label_vertidx, src_flip)):
            if vertidx is not None:
                # if isinstance(vertidx, sparse.csr_matrix):
                #     assert mri_resolution
                #     assert vertidx.shape[1] == stc.data.shape[0]
                #     this_data = np.reshape(stc.data, (stc.data.shape[0], -1))
                #     this_data = vertidx @ this_data
                #     this_data.shape = \
                #         (this_data.shape[0],) + stc.data.shape[1:]
                # else:
                this_data = stc.data[vertidx]
                label_tc[i] = func(flip, this_data)

        # extract label time series for the vol src space (only mean supported)
        offset = nvert[:-n_mean].sum()  # effectively :2 or :0
        for i, nv in enumerate(nvert[2:]):
            if nv != 0:
                v2 = offset + nv
                label_tc[n_mode + i] = np.mean(stc.data[offset:v2], axis=0)
                offset = v2

        # this is a generator!
        yield label_tc

mod_source_estimate._gen_extract_label_time_course=mod_gen_extract_label_time_course
