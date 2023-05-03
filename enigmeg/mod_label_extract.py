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

from mne import source_estimate as mod_source_estimate


from mne.source_estimate import (_pca_flip, _volume_labels,
        _BaseSourceEstimate,_BaseVolSourceEstimate,_BaseVectorSourceEstimate,
        _prepare_label_extraction)



def _pca(flip, data):
    U, s, V = _safe_svd(data, full_matrices=False)
    # determine sign-flip
    # use average power in label for scaling
    scale = np.linalg.norm(s) / np.sqrt(len(data))
    if np.mean(V[0]<0):
        sign=-1
    else:
        sign=1
    return sign * scale * V[0]

_label_funcs = {
    'mean': lambda flip, data: np.mean(data, axis=0),
    'mean_flip': lambda flip, data: np.mean(flip * data, axis=0),
    'max': lambda flip, data: np.max(np.abs(data), axis=0),
    'pca_flip': _pca_flip,
    'pca': _pca
}

def mod_gen_extract_label_time_course(stcs, labels, src, *, mode='mean',
                                   allow_empty=False,
                                   mri_resolution=True, verbose=None):
    # loop through source estimates and extract time series
    from scipy import sparse
    if src is None and mode in ['mean', 'max', 'pca']:
        kind = 'surface'
    else:
        _validate_type(src, SourceSpaces)
        kind = src.kind
    #_check_option('mode', mode, _get_default_label_modes())

    if kind in ('surface', 'mixed'):
        if not isinstance(labels, list):
            labels = [labels]
        use_sparse = False
    else:
        labels = _volume_labels(src, labels, mri_resolution)
        use_sparse = bool(mri_resolution)
    n_mode = len(labels)  # how many processed with the given mode
    n_mean = len(src[2:]) if kind == 'mixed' else 0
    n_labels = n_mode + n_mean
    vertno = func = None
    for si, stc in enumerate(stcs):
        _validate_type(stc, _BaseSourceEstimate, 'stcs[%d]' % (si,),
                       'source estimate')
        #_check_option(
        #    'mode', mode, _get_allowed_label_modes(stc),
        #    'when using a vector and/or volume source estimate')
        if isinstance(stc, (_BaseVolSourceEstimate,
                            _BaseVectorSourceEstimate)):
            mode = 'mean' if mode == 'auto' else mode
        else:
            mode = 'mean_flip' if mode == 'auto' else mode
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)  # avoid keeping a ref
            nvert = np.array([len(v) for v in vertno])
            label_vertidx, src_flip = _prepare_label_extraction(
                stc, labels, src, mode, allow_empty, use_sparse)
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

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

        # do the extraction
        label_tc = np.zeros((n_labels,) + stc.data.shape[1:],
                            dtype=stc.data.dtype)
        for i, (vertidx, flip) in enumerate(zip(label_vertidx, src_flip)):
            if vertidx is not None:
                if isinstance(vertidx, sparse.csr_matrix):
                    assert mri_resolution
                    assert vertidx.shape[1] == stc.data.shape[0]
                    this_data = np.reshape(stc.data, (stc.data.shape[0], -1))
                    this_data = vertidx @ this_data
                    this_data.shape = \
                        (this_data.shape[0],) + stc.data.shape[1:]
                else:
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
