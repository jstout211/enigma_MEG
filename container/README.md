## Make singularity definition file
Neurodocker version 1.0.1
```
neurodocker generate singularity \
    --copy ./enigma_meg.yml ./condastartup.sh / \
    --pkg-manager apt \
    --install git \
    --base-image neurodebian:bullseye\
    --freesurfer version=7.4.1 \
    --miniconda version=latest env_name="enigma_meg" yaml_file="enigma_meg.yml" \
    --entrypoint='/condastartup.sh' \
    --user nonroot > enigma_meg.def 
```

## Build Singularity container
`sudo singularity build enigma_meg.sif enigma_meg.def`

# Running container
`singularity run -B ${BIDS_ROOT}:${BIDS_ROOT} enigma_meg.sif`


