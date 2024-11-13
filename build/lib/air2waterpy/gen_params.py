'''
Util functions for air2water
'''

import numpy as np

# @njit
def get_param_bound(mean_depth, # precomputed depth
                        tw_range = (0, 30),
                        sradmax_range = (200, 450),
                        sradmin_range = (0, 200), 
                        albedo_range = (0.04, 0.2),
                        epsilon_a_range = (0.6, 0.9),
                        alpha_s_range = (3,15),
                        Delta_alpha_s_range = (0.1, 15),
                        ea_range = (5,15),
                        Delta_ea_range = (0.1, 10),
                        ):
    # Original script is written in MATLAB from https://github.com/marcotoffolon/air2water/    
    # This is adapted from its R version at https://github.com/aemon-j/air2wateR/blob/master/R/gen_param.R    
    n = 2
    empty_matrix = np.empty([n+1, n+1])
    Rad_a, Rad_b, ew_par = empty_matrix, empty_matrix, empty_matrix
    
    # Define constants
    rho = 1000 # km/m3
    cp = 4186 # J/kg K
    s_Boltzman = 5.67e-8 # W/m2 K4
    min_epsilon_a = 0.6
    bowen_ratio = 0.61
    
    # Get all possible daily max/min solar radiation, W/m2
    min_maxsrad, max_maxsrad = sradmax_range
    min_minsrad, max_minsrad = sradmin_range
    # delta_maxrad = max_maxsrad - min_maxsrad
    # delta_minrad = max_minsrad - min_minsrad
    maxrad = np.linspace(min_maxsrad, max_maxsrad, num = n+1)
    minrad = np.linspace(min_minsrad, max_minsrad, num = n+1)
    for i in range(n+1):
        for j in range(n+1):
            Rad_a[i,j] = (maxrad[i] - minrad[j]) * 0.5
            Rad_b[i,j] = (minrad[j] + maxrad[i]) * 0.5
    minRad_a, maxRad_a = np.min(Rad_a), np.max(Rad_a) # range of annual amplitude
    minRad_b, maxRad_b = np.min(Rad_b), np.max(Rad_b) # range of annual average
    
    # Plausible range for shortwave reflectivity (albedo) rs
    min_rs, max_rs = albedo_range
    rs = np.linspace(min_rs, max_rs, num = n+1)
    
    # Get all possible water temperature range, C
    min_Temperature_rif, max_Temperature_rif = tw_range
    Temperatura_rif = np.linspace(min_Temperature_rif, max_Temperature_rif, num = n+1)
    Kelvin0 = Temperatura_rif + 273.15
    min_Delta_T, max_Delta_T = tw_range
    Delta_T = np.linspace(min_Delta_T, max_Delta_T, num = n+1)
    
    # Plausible values for the emissivities of atmosphere (epsilon_a) and water (epsilon_w)
    min_epsilon_a, max_epsilon_a = epsilon_a_range
    epsilon_a = 0.97 * np.linspace(min_epsilon_a, max_epsilon_a, num = n+1) # (1-ra)*epsilon_a, where ra=0.03 (reflection of infrared radiation from water surface)
    epsilon_w = 0.97
    
    # Plausible values for the sensible/latent heat transfer functions (alpha_s/alpha_l, W/m2K)
    min_alpha_s, max_alpha_s = alpha_s_range
    alpha_s = np.linspace(min_alpha_s, max_alpha_s, num = n+1)
    alpha_l = alpha_s/bowen_ratio
    min_Delta_alpha_s, max_Delta_alpha_s = Delta_alpha_s_range
    Delta_alpha_s = np.linspace(min_Delta_alpha_s, max_Delta_alpha_s, num = n+1)
    Delta_alpha_l = Delta_alpha_s
    
    # Plausible values for the atmospheric water pressure (ea, mbar)
    min_ea, max_ea = ea_range
    ea = np.linspace(min_ea, max_ea, num = n+1)
    min_Delta_ea, max_Delta_ea = Delta_ea_range
    Delta_ea = np.linspace(min_Delta_ea, max_Delta_ea, num = n+1)
    
    # Plausible values for the water vapor saturation pressure at the temperature of water (ew, mbar)
    for i in range(n+1):
        for j in range(n+1):
            ew_par[i,j] = 6.112*np.exp(Temperatura_rif[i]*17.67/(Temperatura_rif[i]+243.5))
    ew = (np.max([0, np.min(ew_par)]), np.max(ew_par))
    
    # Reaction volume participating to the heat exchange with the atmosphere
    min_D = 1+mean_depth/20
    max_D = np.max([10, mean_depth])
    D = np.linspace(min_D, max_D, num = n+1)
    
    # Denominator
    den = rho * cp * D /86400 # conversino from seconds to days
    
    # parameter 1
    p1_par = np.empty([n+1, n+1, n+1, n+1, n+1, n+1, n+1, n+1], np.float64)
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                for l in range(n+1):
                    for m in range(n+1):
                        for o in range(n+1):
                            for p in range(n+1):
                                for q in range(n+1):
                                    temp = 6.112*np.exp(Temperatura_rif[m]*17.67/(Temperatura_rif[m]+243.5))*(1 - 17.67*243.5/(Temperatura_rif[m]+243.5)**2*Temperatura_rif[m] )
                                    p1_par[i,j,k,l,m,o,p,q] = ( (1-rs[p])*Rad_b[i,i] + s_Boltzman*Kelvin0[m]**3*(epsilon_a[j]-epsilon_w)*(273.15-3*Temperatura_rif[m])+alpha_l[k]*( ea[l] - temp ))/den[o]
                                    
    p1 = (np.min(p1_par), np.min([2, np.max(p1_par)]))
    
    # Parameter 2
    p2 = ((4*s_Boltzman*epsilon_a[0]*Kelvin0[0]**3 + alpha_s[0])/den[n], 
          (4*s_Boltzman*epsilon_a[n]*Kelvin0[n]**3 + alpha_s[n])/den[0] )
    
    # Parameter 3
    p3_par = np.empty([n+1, n+1, n+1, n+1, n+1, n+1, n+1, n+1], np.float64)
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                for l in range(n+1):
                        for o in range(n+1):
                            for p in range(n+1):
                                p3_par[i,j,k,l,o,p] = (4*s_Boltzman*epsilon_a[i]*Kelvin0[o]**3*( 1 - (epsilon_a[i]-epsilon_w)/epsilon_a[i])+alpha_s[p]+alpha_l[j]*6.112*np.exp(Temperatura_rif[k]*17.67/(Temperatura_rif[k]+243.5))*17.67*243.5/(Temperatura_rif[k]+243.5)**2)/den[l]
                                
                                
    p3 = (np.min(p3_par), np.max(p3_par))
    
    # Parameter 4
    p4 = (1, 100 * mean_depth**(-0.35))
    
    # Parameter 5
    p5 = ( ((1-rs[n])*minRad_a + alpha_l[0]*Delta_ea[0]+Delta_alpha_l[0]*(ea[0] - ew[1] + Delta_ea[0]) + Delta_alpha_s[0]*Delta_T[0] )/den[n],
           ((1-rs[0])*maxRad_a + alpha_l[n]*Delta_ea[n]+Delta_alpha_l[n]*(ea[n] - ew[0] + Delta_ea[n]) + Delta_alpha_s[n]*Delta_T[n] )/den[0]
          )
    
    # Parameter 6
    p6 = (0, 1)
    
    # Parameter 7
    p7 = (0, 150)
    
    # Parameter 8
    p8 = (0, 0.5)
    
    return p1, p2, p3, p4, p5, p6, p7, p8

def find_wider_range(bnd1, 
                     bnd2):
    # find low and high boundary
    bnd1_low, bnd1_high = bnd1
    bnd2_low, bnd2_high = bnd2
    # compare low bound
    low_bnd = np.min([bnd1_low, bnd2_low])
    high_bnd = np.max([bnd1_high, bnd2_high])
    
    # update the new bound
    new_bnd = (low_bnd, high_bnd)
    
    return new_bnd
    