
# Installation
## Setup Conda Environment
```#Setup Conda environment
conda create -n enigma_meg python pip
conda activate enigma_meg
```
## Install python package
```#Install enigma meg and dependencies
git clone https://github.com/jstout211/enigma_MEG
pip install ./enigma_MEG
```

# Parameter Heirarchy
```Commandline flags
		Passed Config File
			Default Config File
				Environmental Variables
```

# Setup config file



# Running anatomical process
process_anatomical.py -subjects_dir /data/test_data/SUBJECTS_DIR -subjid ctf_fs -setup_source

# Running meg processing
process_meg.py -subjects_dir /data/test_data/SUBJECTS_DIR -subjid ctf_fs -meg_file /data/test_data/CTF/ctf_rest.ds -er_meg_file /data/test_data/CTF/ctf_eroom.ds -trans /data/test_data/CTF/ctf-trans.fif -line_f 60
