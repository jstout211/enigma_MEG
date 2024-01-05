#!/bin/bash

for i in $(find ./sub-HCP*/ses-1/meg  -name '*eyesopen*.fif' -exec dirname {} \;); do subjid=${i:6:6} ; echo process_meg.py -remove_old -bids_root $(pwd) -subject $subjid -run 01 -session 1 -mains 60 -rest_tag eyesopen -emptyroom_tag empty -n_jobs 6  >> swarm_EO.sh; done
