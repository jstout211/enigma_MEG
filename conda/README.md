# Building for conda
`mamba build -c conda-forge .`

# Local Install 
Add the tarball to the conda env <br>
`mamba create -n enigma_meg` <br>
`conda activate enigma_meg` <br>
`cp conda-bld.tar   ~/miniconda3/envs/enigma_meg/` <br>
`cd  ~/miniconda3/envs/enigma_meg/`<br>
`tar -xvf ----.tar`   #will result in conda-bld folder <br>
`cd ~`<br>
`mamba install --use-local -c conda-forge` <br>
