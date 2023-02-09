# Building for conda
`mamba build -c conda-forge .`

### Currently Erroring while building on computer without graphics card 
`conda.exceptions.InvalidVersionSpec: Invalid version '*=cuda': invalid character(s)`

# Local Install 
Add the tarball to the conda env <br>
`mamba create -n enigma_meg` <br>
`conda activate enigma_meg` <br>
`cp conda-bld.tar   ~/miniconda3/envs/enigma_meg/` <br>
`cd  ~/miniconda3/envs/enigma_meg/`<br>
`tar -xvf conda-bld.tar`   #will result in conda-bld folder <br>
`cd ~`<br>
`mamba install --use-local -c conda-forge` <br>

###  If Errors from mamba 
ImportError: cannot import name 'init_std_stream_encoding' from 'conda.common.compat' (/home/namystam/miniconda3/lib/python3.7/site-packages/conda/common/compat.py) <br>
https://github.com/mamba-org/mamba/issues/1706 <br>
`conda uninstall mamba -y; conda install conda-forge::mamba` <br>
