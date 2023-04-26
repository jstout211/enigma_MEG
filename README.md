# Installation

Setup Conda Environment:

```
conda install --channel=conda-forge --name=base mamba
mamba create --override-channels --channel=conda-forge --name=enigma_meg mne pip 
conda activate enigma_meg
pip install git+https://github.com/jstout211/enigma_MEG.git
```
# Description

The programs in this package perform the full processing pipeline for the ENIGMA BIDS working group. This suite requires
that your data be organized in BIDS format; if you need a tool for that you can use enigma_anonymization_lite. The core tool 
is process_meg.py, which performs all processing steps for the anatomical MRI and the associated MEG. You can either process
a single subject, or you can loop over all subjects in batch mode. In order to do batch processing, you must first run 
parse_bids.py to produce a .csv file manifest of all available MEG scans. Once all the processing is complete, you can generate
QA images using preq_QA.py. Like process_meg.py, prep_QA.py will operate either on a single subject or on all subjects listed
in a the .csv file produced by parse_bids.py. Once the .png files are created, you can use Run_enigma_QA_GUI.py to interactively
label your subject images as good or bad. 

# Main Processing Pipeline
```
usage: process_meg.py [-h] [-bids_root BIDS_ROOT] [-subject SUBJECT] [-subjects_dir SUBJECTS_DIR] [-fs_subject FS_SUBJECT] 
		      [-run RUN] [-session SESSION] [-mains MAINS] [-rest_tag REST_TAG] [-emptyroom_tag EMPTYROOM_TAG] 
		      [-fs_ave_fids] [-proc_fromcsv PROC_FROMCSV] [-n_jobs NJOBS]
		      
This runs all anatomical MRI and MEG processing to produce ENIGMA MEG working gruop results.

optional arguments: 
  -h, --help            show this help message and exit 
  -bids_root BIDS_ROOT 	the root directory for the BIDS tree
  -subjects_dir SUBJECTS_DIR
                        Freesurfer subjects_dir can be assigned at the commandline
                        if not in bids_root/derivatives/freesurfer/subjects
  -subjid SUBJID        Define subjects id 
  -fs_subject FSID	Define subject's freesurfer ID if different from BIDS subject ID
  -meg_file MEG_FILE    Location of meg rest dataset
  -run RUN		Run identifier for MEG scan in BIDS dir
  -session SESSION	Session identifier for MEG scan in BIDS dir
  -mains MAINS		Powerline frequency, defaults to 60
  -rest_tag REST_TAG	Stem for finding resting state MEG scans in the BIDS directory, defaults to 'rest'
  -emptyroom_tag EMPTYROOM_TAG
  			Stem for finding emptyroom datasets in the BIDS directory, defaults to 'emptyroom'
  -fs_ave_fids		optional course registration, please don't use this option unless you have to
  -proc_fromcsv	CSVFILE	Option to loop over rows in the csv file created by parse_bids.py
  -n_jobs NJOBS		Number of workers to use for multithreaded processing
```
Output:
```
BIDS_ROOT/derivatives/ENIGMA_MEG/logs/sub-SUBJID_ses-SESSION_log.txt
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_bem.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_fooof_results_run-01/Band_rel_power.csv
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_fooof_results_run-01/label_spectra.csv
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_run-RUN_lcmv.h5
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_src.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-EMPTYROOM_TAG_run-RUN_cov.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-EMPTYROOM_TAG_run-RUN_epo.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-EMPTYROOM_TAG_run-RUN_proc-filt_meg.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-REST_TAG_run-RUN_cov.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-REST_TAG_run-RUN_epo.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-REST_TAG_run-RUN_fwd.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-REST_TAG_run-RUN_trans.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJID/ses-SESSION/meg/sub-SUBJID_ses-SESSION_task-REST_TAG_run-RUN_proc-filt_meg.fif
```
Example:
```
process_meg.py -subjects_dir /data/freesurfer/subjects -bids_root bids_out -subject sub-Subj01 -fs_subject Sub01 -run 01
	-session 1 -mains 50 -rest_tag rest -emptyroom_tag noise -n_jobs 4
```
This example will run a single subject. This subject has already been processed by freesurfer, and the subjects directory is not
in the BIDS tree. In addition, the freesurfer subject ID and BIDS subject ID are different. The rest MEG images are labeled 
'task-rest' and the emptyroom datasets are labeled 'task-noise' in the BIDS directory. This data was collected in Europe, so the
mains frequency is 50Hz. Process the data collected in session '1' and run '01'. Use 4 workers for any multithreaded functions.
```
process_meg.py -mains 60 -n_jobs 8 -proc_fromcsv ParsedBIDS_dataframe.csv
```
In this example, the data were collected in the US where the mains frequency is 60Hz. Use 8 jobs for multithreaded operations. 
Instead of processing a single subject, process all the subjects in the 'ParsedBIDS_dataframe.csv' file, which was created by 
parse_bids.py

# Parsing the BIDS tree for batch processing
```
usage: parse_bids.py [-h] [-bids_root BIDS_ROOT] [-rest_tag REST_TAG] [-emptyroom_tag EMPTYROOM_TAG]

This python script parses a BIDS directory into a CSV file with one line per MEG to process

options:
  -h, --help            show this help message and exit
  -bids_root BIDS_ROOT  The name of the BIDS directory to be parsed
  -rest_tag REST_TAG    The filename stem to find rest datasets
  -emptyroom_tag EMPTYROOM_TAG
                        The filename stem to find emptyroom datasets
```
Output:
```
ParsedBIDS_dataframe.csv
```
Example:
```
parse_bids.py -bids_root bids_out -rest_tag resteyesopen -emptyroom_tag emptyroom
```
In this example, parse the BIDS tree located in 'bids_out'. The rest MEG images are labeled 'task-resteyesopen' and the 
emptyroom datasets are labeled 'task-emptyroom'. 

# Extract QA images for assessment 
```
usage: enigma_prep_QA.py [-h] [-bids_root BIDS_ROOT] [-subjects_dir SUBJECTS_DIR] [-subjid SUBJID] [-session SESSION] [-run RUN] [-proc_from_csv PROC_FROM_CSV]

This python script will compile a series of QA images for assessment of the enigma_MEG pipeline

options:
  -h, --help            show this help message and exit
  -bids_root BIDS_ROOT  BIDS root directory
  -subjects_dir SUBJECTS_DIR
                        Freesurfer subjects_dir can be assigned at the commandline if not already exported
  -subjid SUBJID        Define the BIDS subject id to process
  -fs_subjid FSID	Freesurfer subject id if different from subjid
  -session SESSION      Session number
  -run RUN              Run number, note that 01 is different from 1
  -proc_from_csv PROC_FROM_CSV
                        Loop over all subjects in a .csv file (ideally produced by parse_bids.py
```
Output:
```
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_beamformer.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_bem.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_coreg.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_spectra.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_src.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_surf.png
```
Example:

```
enigma_prep_QA.py -bids_root BIDS -subjid sub-Subj1 -fs_subjid Subj1 -subjects_dir /home/freesurfer/subjects -session 1 -run 01
```
In this example, the bids directory is called BIDS, and we want to prepare the QA images for subject sub-Sub1. The freesurfer 
processing was run separately in the directory /home/freesurfer/subjects, where the subject id is Sub1. Process the QA images
for session 1 and run01.

# Run the QA GUI 
```
usage: Run_enigma_QA_GUI.py [-h] [-bids_root BIDS_ROOT] [-QAtype QATYPE] [-rows ROWS] [-columns COLUMNS] [-imgsize IMGSIZE]

options:
  -h, --help            show this help message and exit
  -bids_root BIDS_ROOT  Location of bids directory, default=bids_out
  -QAtype QATYPE        QA type to run. Options are:
  			'coreg' - produces images showing MEG/MRI alignment
			'bem' - shows the bem surface overlaid on the anatomical MRI
			'surf' - shows the freesurfer source reconstruction from multiple angles, along with the parcellation
			'src' - shows the source space 
			'spectra' - shows the average power spectral density in all sensors (overlaid)
			'beamformer' - shows the source localized (per parcel) alpha power overlaid on the brain surface
  -rows ROWS            number of rows in QA browser, default value dependent on QAtype
  -columns COLUMNS      number of columns in QA browser, default value dependent on QAtype
  -imgsize IMGSIZE      make images smaller or larger, default value dependent on QAtype
  ```
  Output:
  ```
  BIDS_ROOT/derivatives/ENIGMA_MEG_QA/QATYPE_QA_logfile.txt
  ```
  Example:
  ```
  Run_enigma_QA_GUI.py -bids_root BIDS_OUT -QAtype coreg -rows 3 -columns 6 -imgsize 500
  ```
  In this example, the BIDS root directory is BIDS_OUT, and you want to QA the MRI/MEG coregistrations. You have a very large 
  widescreen monitor, so you can QA a 3x6 array of images at once. 
