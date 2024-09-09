#!/usr/bin/env bash
set -e
export USER="${USER:=`whoami`}"
export PATH=/opt/miniconda-latest/envs/enigma_meg/bin:$PATH
if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi
