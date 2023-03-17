# The below is currently under construction and will be changing 


______________________

# Installation
## Setup Conda Environment
```#Setup Conda environment
conda install --channel=conda-forge --name=base mamba
mamba create --override-channels --channel=conda-forge --name=enigma_meg mne pip 
conda activate enigma_meg
pip install git+https://github.com/jstout211/enigma_MEG.git
```

# Parameter Heirarchy
```
Commandline flags
    Passed Config File
        Default Config File
	    Environmental Variables
```

# Setup config file



# Running anatomical process
```
usage: process_anatomical.py [-h] [-subjects_dir SUBJECTS_DIR] [-subjid SUBJID]
                             [-recon_check RECON_CHECK] [-recon1] [-recon2]
                             [-recon3] [-setup_source] [-run_unprocessed]

Processing for the anatomical inputs of the enigma pipeline

optional arguments:
  -h, --help            show this help message and exit
  -subjects_dir SUBJECTS_DIR
                        Freesurfer subjects_dir can be assigned at the commandline
                        if not already exported.
  -subjid SUBJID        Define subjects id (folder name) in the SUBJECTS_DIR
  -recon_check RECON_CHECK
                        Process all anatomical steps that have not been completed
                        already. This will check the major outputs from autorecon1,
                        2, 3, and mne source setup and proceed with the processing.
                        The default is set to TRUE
  -recon1               Force recon1 to be processed
  -recon2               Force recon2 to be processed
  -recon3               Force recon3 to be processed
  -setup_source         Runs the setup source space processing in mne python to
                        create the BEM model
  -run_unprocessed      Checks for all unrun processes and runs any additional steps
                        for inputs to the source model
```

Example:
```
process_anatomical.py -subjects_dir /data/test_data/SUBJECTS_DIR -subjid ctf_fs -setup_source
```

# Running meg processing
```
usage: process_meg.py [-h] [-subjects_dir SUBJECTS_DIR] [-subjid SUBJID] 
                      [-meg_file MEG_FILE] [-er_meg_file ER_MEG_FILE] [-viz_coreg] 
                      [-trans TRANS] [-line_f LINE_F] 

optional arguments: 
  -h, --help            show this help message and exit 
  -subjects_dir SUBJECTS_DIR
                        Freesurfer subjects_dir can be assigned at the commandline
                        if not already exported.
  -subjid SUBJID        Define subjects id (folder name) in the SUBJECTS_DIR
  -meg_file MEG_FILE    Location of meg rest dataset
  -er_meg_file ER_MEG_FILE
                        Emptyroom dataset assiated with meg file
  -viz_coreg            Open up a window to vizualize the coregistration between
                        head surface and MEG sensors
  -trans TRANS          Transfile from mne python -trans.fif
  -line_f LINE_F        Line frequecy
```

Example:
```
process_meg.py -subjects_dir /data/test_data/SUBJECTS_DIR -subjid ctf_fs -meg_file /data/test_data/CTF/ctf_rest.ds -er_meg_file /data/test_data/CTF/ctf_eroom.ds -trans /data/test_data/CTF/ctf-trans.fif -line_f 60
```
