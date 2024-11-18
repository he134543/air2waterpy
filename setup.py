from setuptools import setup, find_packages

long_description = "A python pacakge implementing an air2water model. Numba is used to speed up the computation. Parrallel computing is included for the calibration process. For detailed examples, check our Github repository: https://github.com/he134543/air2waterpy"

setup(
    name="air2waterpy",
    version="0.0.2",
    author="Xinchen He",
    author_email="xinchenhe@umass.edu",
    description="A python pacakge for running the air2water model",
    long_description= long_description,
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