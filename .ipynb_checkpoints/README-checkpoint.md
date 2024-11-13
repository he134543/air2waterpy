# air2waterpy
**air2waterpy** is a Python package implementing the air2water model (Piccolroaz et al., 2013), a lump model for simulating lake surface water temperature (LSWT) based on air temperature. The original air2water model is written in Fortran, [link to the repo](https://github.com/marcotoffolon/air2water). In this pacakge, we rewrote the model code with [numpy](https://numpy.org/) and [numba](https://numba.pydata.org/) which can allow users who are more familar with python to implement an air2water model in few lines of code. The code structure is adapted from the Rainfall-Runoff modelling playground ([RRMPG](https://github.com/kratzert/RRMPG)).


# Main features
Main features of this python package:
- Used [Numba](https://numba.pydata.org/) was used for python code acceleration.
- Applied a fourth-order Runge-Kutta(RK4) method for approximating the numerical solution of the ODE system.
- Employed the [pyswarms](https://pyswarms.readthedocs.io/en/latest/) to calibrate the air2water model.
- Support parallel computation to speed up the model calibration.

# Data preparation

- Daily air temperature: needs to be daily continuous to drive the daily model.
- Daily LSWT: for calibration/validation.

# Installation

```{bash}
pip install air2waterpy
```

# Quick Start

```python
import pandas as pd
from air2waterpy import air2water
import matplotlib.pyplot as plt

# Load air temperature and water temperature data of Lake Superior
df = pd.read_csv("test/superior.csv")

# test api of update parameter boundary based on the depth
model.update_param_bnds(mean_depth_range=(140, 150))

# initialize a model
model = air2water(version = "8p")

# select calibration period
cal_period = pd.date_range("1995-01-01", "2004-01-01")
val_period = pd.date_range("2005-01-01", "2011-12-31")
cal_df = df.loc[cal_period]
val_df = df.loc[val_period]

# particle swarm optimization for calibration
# use 10 cpus for parallel calibration
cost, joint_vars = model.pso_fit(cal_df.tw.to_numpy(), cal_df.ta, cal_period, n_cpus=10)

# load parameters
model.load_params(dict(zip(model._param_list, joint_vars)))

# plot the simulation performance
plt.scatter(val_period, val_df.tw)
plt.plot(val_tw_sim, color = "k")
```