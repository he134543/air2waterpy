from setuptools import setup, find_packages

setup(
    name="air2waterpy",
    version="0.0.1",
    author="Xinchen He",
    author_email="xinchenhe@umass.edu",
    description="A_python_pacakge_for_running_the_air2water_model",
    long_description="**air2waterpy** is a Python package implementing the air2water model (Piccolroaz et al., 2013), a lump model for simulating lake surface water temperature (LSWT) based on air temperature. The original air2water model is written in Fortran, [link to the repo](https://github.com/marcotoffolon/air2water). In this pacakge, we rewrote the model code with [numpy](https://numpy.org/) and [numba](https://numba.pydata.org/) which can allow users who are more familar with python to implement an air2water model in few lines of code. The code structure is adapted from the Rainfall-Runoff modelling playground ([RRMPG](https://github.com/kratzert/RRMPG)).",
    url="https://github.com/he134543/air2waterpy",
    packages=find_packages(),
    install_requires =[
        'numpy>=2.0.2',
        'pandas>=2.2.3',
        'pyswarms>=1.3.0',
        'numba>=0.60.0',
        'joblib>=1.4.2'
        ],
    license="MIT-License",
    classifiers=[
          'Programming Language :: Python :: 3.12',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering',
          'Intended Audience :: Science/Research'
        ]
    )