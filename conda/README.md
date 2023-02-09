# Building for conda
`mamba build -c conda-forge .`

# Local Install 
Add the tarball to the conda env <br>
`mamba create -n engima_meg` <br>
`cp ----.tar   ~/miniconda3/envs/enigma_meg/` <br>
`tar -xvf ----.tar`   #will result in conda-bld folder <br>
`mamba install --use-local -c conda-forge` <br>
