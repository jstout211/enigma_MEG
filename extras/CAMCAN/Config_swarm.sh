#export ENIGMA_REST_DIR=$(pwd)/camcan_enigma_outputs
#export SUBJECTS_DIR=$(pwd)/Camcan_fs


subjid=sub-CC210250; ~/src/enigma/python_code/process_meg.py -subjid ${subjid} -meg_file camcan_mne_bids/${subjid}/meg/${subjid}_ses-rest_task-rest_proc-sss.fif -er_meg_file camcan_mne_bids/${subjid}/meg/emptyroom_*.fif -trans camcan_mne_bids/${subjid}/anat/${subjid}-trans.fif -line_f 50

## Change the below to have full paths $(pwd)/camcan_mne_bids ...

cd /data/EnigmaMeg/camcan_mne_bids
subjids=$(ls -d sub-CC*)

topdir=/data/EnigmaMeg
cd $topdir
export ENIGMA_REST_DIR=${topdir}/camcan_enigma_outputs
export SUBJECTS_DIR=${topdir}/Camcan_fs



for subjid in $subjids; do 
if [ -f ${topdir}/camcan_mne_bids/${subjid}/meg/emptyroom_*.fif ]; then
echo ~/src/enigma/python_code/process_meg.py -subjid ${subjid} -meg_file ${topdir}/camcan_mne_bids/${subjid}/meg/${subjid}_ses-rest_task-rest_proc-sss.fif -er_meg_file ${topdir}/camcan_mne_bids/${subjid}/meg/emptyroom_*.fif -trans ${topdir}/camcan_mne_bids/${subjid}/anat/${subjid}-trans.fif -line_f 50 >> swarm_meg_proc.sh; 
fi;
done


swarm -f swarm_meg_proc.sh -g 6 -t 4 -b 6 --logdir old_procs/process_meg
