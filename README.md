
# Installation
## Setup Conda Environment
```#Setup Conda environment
conda create -n enigma_meg
conda activate enigma_meg
conda install python pip
```
## Install python package
```#Install enigma meg and dependencies
git clone https://github.com/jstout211/enigma_MEG
pip install ./enigma_MEG
```

# Setup config file


# Running anatomical process
./process_anatomical.py -subjects_dir /data/test_data/SUBJECTS_DIR -subjid ctf_fs -setup_source

# Running meg processing
./process_meg.py -subjects_dir /data/test_data/SUBJECTS_DIR -subjid ctf_fs -meg_file /data/test_data/CTF/ctf_rest.ds -er_meg_file /data/test_data/CTF/ctf_eroom.ds -trans /data/test_data/CTF/ctf-trans.fif -line_f 60
