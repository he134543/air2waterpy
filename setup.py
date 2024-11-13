from setuptools import setup, find_packages

setup(
    name="air2waterpy",
    version="0.0.1",
    author="Xinchen He",
    author_email="xinchenhe@umass.edu",
    description=("A python pacakge for running the air2water model"),
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