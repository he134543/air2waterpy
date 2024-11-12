# air2waterpy
A python package of the air2water model (Piccolroaz et al, 2013), a lumped model that simulates lake surface water temperature (LSWT) using only air temperature. The original air2water model is written in Fortran [here](https://github.com/marcotoffolon/air2water). In this pacakge, the source code of the model is rewritten in python. Note the speed of the python is not comparable to Fortran, but we used [numpy](https://numpy.org/) and [numba](https://numba.pydata.org/) to accelerate the performance. It might still not be as fast as the Fortran version, this aims to provide a simple interface for python users to quickly implement the model without worrying about configuring the Fortran environment.

# Data preparation

- Daily air temperature: needs to be daily continuous to drive the daily model.
- Daily LSWT: for calibration/validation.

# Example usage