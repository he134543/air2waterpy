'''
This module contains functions to calculate evaluation metrics
'''

import numpy as np
from scipy.stats.stats import pearsonr


def calc_mse(obs, sim):
    """Calculate the mean squared error.
    Args:
        obs: Array of the observed values
        sim: Array of the simulated values
    Returns:
        The MSE value for the simulation, compared to the observation.
    """
    # Validation check on the input arrays
    # drop nan and negative temperature from the observation    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
    obs = obs[obs>=0]
    sim = sim[obs>=0]
    
    # Calculate the rmse value
    mse_val = np.mean((obs-sim)**2)

    return mse_val

def calc_nse(obs, sim):
    """Calculate the Nash-Sutcliffe model efficiency coefficient.
    Args:
        obs: Array of the observed values
        sim: Array of the simulated values
    Returns:
        The NSE value for the simulation, compared to the observation.
    """
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
    # denominator of the fraction term
    denominator = np.sum((obs-np.mean(obs))**2)
    
    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        raise ValueError("NSE denominator is 0")
    
    # numerator of the fraction term
    numerator = np.sum((sim-obs)**2)

    # calculate the NSE
    nse = 1 - numerator/denominator
    
    return nse