# Building for conda
`mamba build -c conda-forge .`

# Local Install 
Add the tarball to the 
`mamba create -n engima_meg`
`cp ----.tar   ~/miniconda3/envs/enigma_meg/`
`tar -xvf ----.tar`   #will result in conda-bld folder
`mamba install --use-local -c conda-forge`
