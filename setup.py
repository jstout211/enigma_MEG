import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enigma", 
    version="0.2",
    author="Jeff Stout",
    author_email="stoutjd@nih.gov",
    description="Package to calculate metrics for the Enigma MEG consortium",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jstout211/enigma",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    python_requires='<3.12',
    install_requires=['mne', 'numpy', 'scipy', 'pandas', 'neurodsp', 'fooof', 'munch', 'pysimplegui<5.0',
                      'statsmodels', 'nibabel', 'pytest', 'joblib', 'seaborn', 'mne_bids','MEGnet @ git+https://github.com/nih-megcore/MegNET_2020.git', 'pyctf-lite @ git+https://github.com/nih-megcore/pyctf-lite.git' ], #'pyvista', 'pyqt5','pyvistaqt',
    extras_require={"testing":['datalad','pytest','pygit2']},
    scripts=['enigmeg/process_meg.py', 
             'enigmeg/process_anatomical.py',
             'enigmeg/QA/enigma_prep_QA.py',
	     'enigmeg/QA/Run_enigma_QA_GUI.py',
	     'enigmeg/parse_bids.py', 'extras/copy_enigma_tree.py'],
    )
