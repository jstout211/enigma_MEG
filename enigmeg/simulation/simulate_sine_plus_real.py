
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import find_events, Epochs, compute_covariance, make_ad_hoc_cov
from mne.datasets import sample
from mne.simulation import (simulate_sparse_stc, simulate_raw,
                            add_noise, add_ecg, add_eog)


# sfreq=300
# duration=300

def data_fun(times, amplitude=1, freq=10):
    """Generate sinusoidal data at amplitude (in nAm)"""
    data = amplitude * 1e-9 * np.sin(2. * np.pi * freq * times)
    return data

from functools import partial
def generate_combined_simulation(raw_fname, 
                                 fwd, 
                                 subject=None,
                                 subjects_dir=None,
                                 topdir=None,
                                 label_index=None,
                                 sine_amplitude=None,
                                 sine_frequency=None):
    """Create a combined dataset of simulated plus real data and save
    to the topdir/(subjid)_(AMP)_nAm_(HZ)_hz folder"""
    
    os.chdir(topdir)
    if label_index==None:
        print('Must provide a label index for simulation')
        exit(1)
    
    raw = mne.io.read_raw_fif(raw_fname)
    rng = np.random.RandomState(0)  # random state (make reproducible)
    
    #Labels for simulation
    labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir )
    labels_sim = [labels[label_index]]
    
    times = raw.times #[:int(raw.info['sfreq'] * epoch_duration)]
    src = fwd['src']

    sig_generator = partial(data_fun, amplitude=sine_amplitude, 
                            freq=sine_frequency)    
    
    stc = simulate_sparse_stc(src, n_dipoles=1, times=times,
                              data_fun=sig_generator, labels=labels_sim, 
                              location='center', subjects_dir=subjects_dir)  
    
    # Simulate raw data
    raw_sim = simulate_raw(raw.info, [stc] * 1, forward=fwd, verbose=True)

    #Load raw and save to outfolder
    raw.load_data()

    #Combine simulation w/raw
    comb_out_fname='{}_{}_label_{}_nAm_{}_hz_meg.fif'.format(subject, 
                                                             str(label_index), 
                                                             sine_amplitude, 
                                                             sine_frequency)
    # comb_out_fname = op.join(outfolder, outfolder+'_meg.fif')
    combined = raw.copy()
    combined._data += raw_sim.get_data()
    combined.save(comb_out_fname)
    print('Saved {}'.format(comb_out_fname))
    
    #Save stc for later use
    stc_out_fname = op.join('{}_{}_label_{}_nAm_{}_hz-stc.fif'.format(subject, 
                                                             str(label_index),
                                                             sine_amplitude, 
                                                             sine_frequency))
    stc.save(stc_out_fname)
    
@pytest.mark.sim
def test_iterate_elekta_simulations():
    from enigmeg.test_data.get_test_data import datasets
    
    topdir='/home/stoutjd/data/NEWTEST'
    
    #For elekta data
    elekta_dat = datasets().elekta
    
    subject = 'sub-CC320342' #elekta_dat['subject']
    subjects_dir = '/data/CAMCAN/SUBJECTS_DIR' #elekta_dat['SUBJECTS_DIR']
    raw_fname = elekta_dat['meg_rest']
    eroom_fname = elekta_dat['meg_eroom']
    trans_fname = elekta_dat['trans']
    src_fname = elekta_dat['src']
    bem_fname = elekta_dat['bem']
    input_dir = elekta_dat['enigma_outputs'] 
    
    raw=mne.io.read_raw_fif(raw_fname)
    fwd = mne.forward.make_forward_solution(raw.info, trans_fname, src_fname, 
                                            bem_fname)
    
    import itertools
    amp_vals=np.arange(10, 100, 10) 
    freq_vals=np.arange(5, 20, 5) 
    label_index = np.arange(0,68) #For DK atlas
    
    amp_freq_label = itertools.product(amp_vals, freq_vals, label_index)
    
    output_dir = op.join(topdir, subject)
    if not op.exists(output_dir):
        os.mkdir(output_dir)
        
    #Load raw and save to outfolder
    raw.load_data()
    raw.resample(300)
    raw_out_fname = op.join(output_dir, '{}_raw_meg.fif'.format(subject))
    raw.save(raw_out_fname, overwrite=True)       
        
    for amp,freq,label_idx in amp_freq_label: 
        generate_combined_simulation(raw_out_fname, 
                                 fwd, 
                                 subject=subject,
                                 subjects_dir=subjects_dir,
                                 topdir=output_dir,
                                 label_index=label_idx,
                                 sine_amplitude=amp,
                                 sine_frequency=freq)

        
    





