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
        'numba>=0.60.0'
        ],
    license="MIT-License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"]
    )