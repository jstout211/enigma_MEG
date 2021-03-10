import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enigma", 
    version="0.0.1",
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
    python_requires='>=3.6',
    install_requires=['mne', 'numpy', 'scipy', 'pandas', 'neurodsp', 'fooof',
                      'statsmodels', 'nibabel', 'pytest', 'joblib'],
    scripts=['enigmeg/process_meg.py', 
             'enigmeg/process_anatomical.py'],
)
