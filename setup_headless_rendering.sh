sinteractive --mem=6G --cpus-per-task=4 --gres=lscratch:10

export MESA_GL_VERSION_OVERRIDE=3.3
module load mesa
#conda activate mne

export XDG_RUNTIME_DIR=/lscratch/$SLURM_JOB_ID
Xvfb :1 -screen 0 1280x1024x24 -auth localhost&
export DISPLAY=:1

#python -c 'from mayavi import mlab; mlab.options.offscreen = True; mlab.test_contour3d(); mlab.savefig("./example4.png")' 

# if [ $(hostname) == *.hpc.nih.gov ]

export ENIGMA_REST_DIR=$(pwd)/camcan_enigma_outputs;   
export SUBJECTS_DIR=$(pwd)/Camcan_fs; 

~/src/enigma/python_code/process_meg.py -subjid sub-CC321506  -viz_coreg -meg_file camcan_mne_bids/sub-CC321506/meg/sub-CC321506_ses-rest_task-rest_proc-sss.fif  -trans camcan_mne_bids/sub-CC321506/anat/sub-CC321506-trans.fif -subjects_dir Camcan_fs

~/src/enigma/python_code/process_meg.py -subjid sub-CC321506  -viz_coreg -meg_file camcan_mne_bids/sub-CC321506/meg/sub-CC321506_ses-rest_task-rest_proc-sss.fif  -trans camcan_mne_bids/sub-CC321506/anat/sub-CC321506-trans.fif -subjects_dir Camcan_fs



