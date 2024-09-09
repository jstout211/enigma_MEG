# !!Under Construction!!
TODO: 
  Link subjects dir to an external location <br>
  Confirm python root change works <br>

## Make singularity definition file
Neurodocker version 1.0.1
```
neurodocker generate singularity \
    --copy ./enigma_meg.yml /enigma_meg.yml \
    --pkg-manager apt \
    --install git\
    --base-image neurodebian:bullseye\
    --freesurfer version=7.4.1 \
    --miniconda version=latest env_name="enigma_meg" yaml_file="enigma_meg.yml"\
    --user nonroot > enigma_meg.def``
```
## Change python root in def file
`sed -i 's#/opt/miniconda-latest/bin#/opt/miniconda-latest/envs/enigma_meg/bin#g'`

## Build Singularity container
`sudo singularity build enigma_meg.sif enigma_meg.def`

# Running container
`singularity run -B ${BIDS_ROOT}:${BIDS_ROOT} enigma_meg.sif`


