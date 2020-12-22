#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:06:22 2020

@author: stoutjd
"""
import os
import os.path as op

class datasets():
    def __init__(self):
        basedir = op.join(op.dirname(__file__), 'test_data')
        
        print('HCP')
        print('Need to make HCP MNE src/bem/tran files - using CTF fillins')
        verify_inputs(get_hcp(basedir))
        self.hcp = get_hcp(basedir) 
        
        print('Elekta')
        verify_inputs(get_elekta(basedir))
        self.elekta = get_elekta(basedir)
        
        print('CTF')
        verify_inputs(get_ctf(basedir))
        self.ctf = get_ctf(basedir)
        # self.elekta = get_ctf(basedir, inputs=defaults)
        

def get_hcp(topdir=None):
    inputs=dict()
    inputs['meg_rest'] = op.join(topdir, 'HCP', 'hcp_rest_example.fif')
    inputs['meg_eroom'] = op.join(topdir, 'HCP', 'hcp_eroom_example.fif')
    inputs['enigma_outputs'] = op.join(topdir, 'enigma_outputs')
    inputs['trans'] = op.join(topdir, 'CTF', 'ctf-trans.fif')
    inputs['src'] = op.join(inputs['enigma_outputs'], 'ctf_fs','source_space-src.fif')
    inputs['bem'] = op.join(inputs['enigma_outputs'], 'ctf_fs', 'bem_sol-sol.fif')
    return inputs

def get_ctf(topdir=None):
    inputs = dict() 
    inputs['meg_rest'] = op.join(topdir, 'CTF', 'ctf_rest.ds')
    inputs['meg_eroom'] = op.join(topdir, 'CTF', 'ctf_eroom.ds')
    inputs['enigma_outputs'] = op.join(topdir, 'enigma_outputs')
    inputs['trans'] = op.join(topdir, 'CTF', 'ctf-trans.fif')
    inputs['src'] = op.join(inputs['enigma_outputs'], 'ctf_fs', 'source_space-src.fif')
    inputs['bem'] = op.join(inputs['enigma_outputs'], 'ctf_fs', 'bem_sol-sol.fif')
    return inputs

def get_elekta(topdir=None):
    inputs = dict() 
    inputs['meg_rest'] = op.join(topdir, 'HCP', 'hcp_rest_example.fif')
    inputs['meg_eroom'] = op.join(topdir, 'HCP', 'hcp_eroom_example.fif')
    inputs['enigma_outputs'] = op.join(topdir, 'enigma_outputs')
    inputs['trans'] = op.join(topdir, 'CTF', 'ctf-trans.fif')
    inputs['src'] = op.join(inputs['enigma_outputs'], 'elekta_fs', 'source_space-src.fif')
    inputs['bem'] = op.join(inputs['enigma_outputs'], 'elekta_fs', 'bem_sol-sol.fif')
    return inputs

def verify_inputs(testset):
    '''Loop over all keys in dictionary and verify that datasets are present'''
    found=[]
    found_keys=[]
    notfound=[]
    notfound_keys=[]
    for key in testset.keys():
        if os.path.exists(testset[key]):
            found.append(testset[key])
            found_keys.append(key)
        else:
            notfound.append(testset[key])
            notfound_keys.append(key)
    if len(notfound_keys)==0:
        print('All datasets found and loaded in dictionary')
        print(found_keys)
    else:
        print('{} items found, {} items not found'.format(str(len(found_keys)), str(len(notfound_keys))))
        for notval in notfound:
            print('Not found: {}'.format(notval))
    print()
        
        
