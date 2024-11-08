# -*- coding: utf-8 -*-
# This content of this code is designed based on the https://www.pagepressjournals.org/aiol/article/view/5791/5092
# The format is designed based on the RRMPG: https://github.com/kratzert/RRMPG/

import numpy as np

from numba import njit

@njit()
def _a2w(a1, a2, a3, a4, a5, a6, Ta, Tw, T_Ty, Th):
    # calculate the stratification first
    if Tw >= Th:
        delta = np.exp(-(Tw - Th)/a4)
        if delta == 0:
            delta = 1e-3
        
    else:
        delta = 1
    
    # delta has to be larger than 0
    k = (a1 + a2 * Ta - a3 * Tw + a5 * np.cos(2 * np.pi * (T_Ty - a6) ))/delta
    
    # print(Tw, Th, a4)

    return k


@njit()
def run_air2water6p(ta, 
                    t_ty,
                    th, 
                    tw_init, 
                    tw_ice,
                    # num_mod,
                    params):
    """Implementation of the GR4J model.
    
    This function should be called via the .simulate() function of the air2water6p
    class and not directly. It is kept in a separate file for less confusion
    if anyone wants to inspect the actual model routine.
    
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
    
    # Unpack the model parameters
    a1 = params['a1']
    a2 = params['a2']
    a3 = params['a3']
    a4 = params['a4']
    a5 = params['a5']
    a6 = params['a6']
    
    # initialize empty arrays for lake surface water temperature
    tw = np.zeros(num_timesteps, np.float64)
    
    # set initial values
    tw[0] = tw_init
    
    # Start the model simulation loop
    # Use the forward RK45 explicit solution
    # time step dt = 1
    
    # if num_mod == "RK45":    
    
    for t in range(1, num_timesteps):
        
        # try:      
        k1 = _a2w(a1, a2, a3, a4, a5, a6, 
                    ta[t-1], tw[t-1], t_ty[t-1], th)
        
        k2 = _a2w(a1, a2, a3, a4, a5, a6, 
                    (ta[t-1] + ta[t])/2, tw[t-1] + 0.5 * k1, (t_ty[t-1] + t_ty[t])/2, th)
        
        k3 = _a2w(a1, a2, a3, a4, a5, a6, 
                    (ta[t-1] + ta[t])/2, tw[t-1] + 0.5 * k2, (t_ty[t-1] + t_ty[t])/2, th)
        
        k4 = _a2w(a1, a2, a3, a4, a5, a6, 
                    ta[t], tw[t-1] + k3, t_ty[t], th)
        
        tw[t] = tw[t-1] + 1/6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # except:
        #     # this situation often happen when the paramter is not right
        #     # the a4 leads to a break of delta
        #     # raise ValueError(f"{delta}")
        #     # tw[t] = tw[t-1]

        # set the lower bound of the temperature when ice covered
        tw[tw < tw_ice] = tw_ice

    # print("successfully route once")
    
    # return all but the artificial 0's step
    return tw