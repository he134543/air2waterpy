'''
The numba code for running the air2water model

'''

import numpy as np
from numba import njit

@njit()
def _a2w(a1, a2, a3, a4, a5, a6, 
         a7, a8,
         Ta, Tw, T_Ty, Th):
    # step wise calculate the temperature increment
    if Tw >= Th:
        delta = np.exp(-(Tw - Th)/a4)
        if delta == 0:
            delta = 1e-3
    else:
        if np.logical_and(a7 == 0, a8 == 0):
            delta = 1
        else:
            delta = np.exp(-(Th - Tw)/a7) + np.exp(-Tw/a8)
    # delta has to be larger than 0
    k = (a1 + a2 * Ta - a3 * Tw + a5 * np.cos(2 * np.pi * (T_Ty - a6) ))/delta

    return k

@njit()
def run_air2water(ta, 
                  t_ty,
                  th, 
                  tw_init, 
                  tw_ice,
                  model_version,
                  params):
    """Implementation of the air2water model.
    
    
    The naming of the variables is kept as in the original publication [1].
    
    Args:
        tq: Numpy [t] array, which contains the airtemperature input.
        t_ty: Numpy [t] array, which contains the fraction of current day number to the total day number of the year
        th: Scalar for the deep water temperature
        tw_init: Scalar for the initial state of the water temperature.
        params: Numpy array of custom dtype, which contains the model parameter.
        
    Returns:
        tw: Numpy [t] array with the simulated streamflow.
            
    [1] Prediction of lake surface temperature using the air2water model: 
    guidelines, challenges, and future perspectives, 
    Advances in Oceanography and Limnology, 7:36-50, DOI: http://dx.doi.org/10.4081/aiol.2016.5791
        
    """    
    # Number of simulation timesteps
    num_timesteps = len(ta)
    
    # Unpack the model parameters based on the model version
    if model_version == "6p":
        a1 = params['a1']
        a2 = params['a2']
        a3 = params['a3']
        a4 = params['a4']
        a5 = params['a5']
        a6 = params['a6']
        a7 = np.float64(0.0)
        a8 = np.float64(0.0)
    elif model_version == "8p":
        a1 = params['a1']
        a2 = params['a2']
        a3 = params['a3']
        a4 = params['a4']
        a5 = params['a5']
        a6 = params['a6']
        a7 = params['a7']
        a8 = params['a8']
    else:
        print("Select correct model version. '6p' or '8p'")
        return
    # initialize empty arrays for lake surface water temperature
    tw = np.zeros(num_timesteps, np.float64)
    
    # set initial values
    tw[0] = tw_init
    
    # Start the model simulation
    # Use the forward RK45 explicit solution
    # time step dt = 1 day
    for t in range(1, num_timesteps):
        k1 = _a2w(a1, a2, a3, a4, a5, a6, a7, a8,
                    ta[t-1], tw[t-1], t_ty[t-1], th)
        
        k2 = _a2w(a1, a2, a3, a4, a5, a6, a7, a8,
                    (ta[t-1] + ta[t])/2, tw[t-1] + 0.5 * k1, (t_ty[t-1] + t_ty[t])/2, th)
        
        k3 = _a2w(a1, a2, a3, a4, a5, a6, a7, a8,
                    (ta[t-1] + ta[t])/2, tw[t-1] + 0.5 * k2, (t_ty[t-1] + t_ty[t])/2, th)
        
        k4 = _a2w(a1, a2, a3, a4, a5, a6, a7, a8,
                    ta[t], tw[t-1] + k3, t_ty[t], th)
        
        tw[t] = tw[t-1] + 1/6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # set the lower bound of the temperature when ice covered
        tw[tw < tw_ice] = tw_ice
    
    # return all but the artificial 0's step
    return tw