import numbers

import numpy as np

from scipy import optimize
from pyswarms.single.global_best import GlobalBestPSO
from multiprocessing import Pool
from joblib import Parallel, delayed
from functools import partial
from .basemodel import BaseModel
from .air2water6p_model import run_air2water6p
from ..utils.metrics import calc_mse
from ..utils.array_checks import check_for_negatives, validate_array_input
from ..utils.gen_param import get_param_bound, find_wider_range


class air2water6p(BaseModel):
    """Interface to the the ABC-Model.

    This model implements the 6-parameter version of air2water model
    
    The 6-parameter model assume the the delta is 1 when the lake is inversely stratified

    Original Publication:
    Piccolroaz S. (2016), 
    Prediction of lake surface temperature using the air2water model: 
    guidelines, challenges, and future perspectives, Advances in Oceanography and Limnology, 
    7:36-50, DOI: http://dx.doi.org/10.4081/aiol.2016.5791

    Args:
        params: (optional) Dictonary containing all model parameters as a
            seperate key/value pairs.

    """

    # List of model parameters
    _param_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

    # Dictionary with default parameter bounds
    _default_bounds = {'a1': (1e-4, 2),
                       'a2': (1e-4, 0.5),
                       'a3': (1e-4, 0.5),
                       'a4': (1, 50),
                       'a5': (1e-4, 1.2),
                       'a6': (1e-4, 1)}

    # Custom numpy datatype needed for numba input
    _dtype = np.dtype([('a1', np.float64),
                       ('a2', np.float64),
                       ('a3', np.float64),
                       ('a4', np.float64),
                       ('a5', np.float64),
                       ('a6', np.float64)])

    def __init__(self,
                 params=None):
        """Initialize an air2water model

        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.

        """
        super().__init__(params=params)
        
    def update_param_bnds(self, 
                        mean_depth_range = (1,100),
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
        """_Input lake characteristics and change the parameter bound_

        Args:
            mean_depth_range (_float_): _Possible range of the estimated depth_
            tw_range (tuple, optional): _Range of all possible temperature_. Defaults to (0, 30).
            sradmax_range (tuple, optional): _Range of daily maximum shortwave solar radiation_. Defaults to (200, 450).
            sradmin_range (tuple, optional): _Range of daily minimum shortwave solar radiation_. Defaults to (0, 200).
            albedo_range (tuple, optional): _Range of daily minimum shortwave solar radiation_. Defaults to (0.04, 0.2).
            epsilon_a_range (tuple, optional): _Range of epsilon a_. Defaults to (0.6, 0.9).
            alpha_s_range (tuple, optional): _description_. Defaults to (3,15).
            Delta_alpha_s_range (tuple, optional): _description_. Defaults to (0.1, 15).
            ea_range (tuple, optional): _description_. Defaults to (5,15).
            Delta_ea_range (tuple, optional): _description_. Defaults to (0.1, 10).
        """
        mean_depth_low, mean_depth_high = mean_depth_range
        # Calculate the parameter bnds based on different depth estimate
        bnds_deep = get_param_bound(mean_depth=mean_depth_high, 
                                    tw_range=tw_range, 
                                    sradmax_range=sradmax_range, 
                                    sradmin_range=sradmin_range, 
                                    albedo_range=albedo_range,
                                    epsilon_a_range=epsilon_a_range,
                                    alpha_s_range=alpha_s_range,
                                    Delta_alpha_s_range=Delta_alpha_s_range,
                                    ea_range=ea_range,
                                    Delta_ea_range=Delta_ea_range
                                    )
        bnds_shallow = get_param_bound(mean_depth=mean_depth_low,
                                    tw_range=tw_range, 
                                    sradmax_range=sradmax_range, 
                                    sradmin_range=sradmin_range, 
                                    albedo_range=albedo_range,
                                    epsilon_a_range=epsilon_a_range,
                                    alpha_s_range=alpha_s_range,
                                    Delta_alpha_s_range=Delta_alpha_s_range,
                                    ea_range=ea_range,
                                    Delta_ea_range=Delta_ea_range)
        
        # Choose the widest range of each parameter
        for i in range(6):
            self._default_bounds[self._param_list[i]] = find_wider_range(bnds_deep[i], bnds_shallow[i])

    def simulate(self, 
                 ta,
                 t_ty,
                 th = 4.0,
                 tw_init = 1.0,
                 tw_ice = 0.0, 
                 params=None):
        """Simulate the lake surface water temperature for the passed air temperature.

        This function makes sanity checks on the input and then calls the
        externally defined air2water_6p-Model function.

        Args:
            ta: air temperature data for each timestep. Can be a List, numpy
                array or pandas.Series
            t_ty: the fraction of the current day of year to the total number of days of the year.
                Can be a List, numpy array or pandas Series
            th: deep water temperature. Const
            tw_init: (optional) Initial value for the lake surface water temperature.
            params: (optional) Numpy array of parameter sets, that will be 
                evaluated a once in parallel. Must be of the models own custom
                data type. If nothing is passed, the parameters, stored in the 
                model object, will be used.

        Returns:
            An array with the simulated lake surface temperature for each timestep and
            optional an array with the simulated storage.

        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.

        """
        # Validation check of the air temperature input
        ta = validate_array_input(ta, np.float64, 'airtemp')

        # Validation check of the initial state
        if not isinstance(tw_init, numbers.Number):
            msg = ["The variable 'tw_init' must be a numercial scaler "]
            raise TypeError("".join(msg))

        # Cast initial temperature as float
        tw_init = float(tw_init)
        
        # If no parameters were passed, prepare array w. params from attributes
        if params is None:
            params = np.zeros(1, dtype=self._dtype)
            for param in self._param_list:
                params[param] = getattr(self, param)
        
        # Else, check the param input for correct datatype
        else:
            if params.dtype != self._dtype:
                msg = ["The model parameters must be a numpy array of the ",
                       "models own custom data type."]
                raise TypeError("".join(msg))
            # if only one parameter set is passed, expand dimensions to 1D
            if isinstance(params, np.void):
                params = np.expand_dims(params, params.ndim)
        
        # Create output arrays
        tw = np.zeros((ta.shape[0], params.size), np.float64)
    
        # call simulation function for each parameter set
        for i in range(params.size):
        # Call air2water6p simulation function and return results
            tw[:,i] = run_air2water6p(ta, t_ty, th, tw_init, tw_ice, params[i])

        return tw

    def fit(self, 
            tw_obs, 
            ta,
            t_ty,
            th = 4.0,
            tw_init = 1.0,
            tw_ice = 0.0,
            ):
        """Fit the model to a timeseries of lake water temperature using.

        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed 
        lake water temperature is simulated as good as possible.

        Args:
            tw_obs: Array of observed lake water temperature.
            ta: Array of air temperature data.
            t_ty: Array of day of year/ total number of day of year
            th: (optional) deep water temperature
            tw_init: (optional) Initial value for the water temperature.

        Returns:
            res: A scipy OptimizeResult class object.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.

        """
        # Validation check of the inputs
        tw_obs = validate_array_input(tw_obs, np.float64, 'tw_obs')
        ta = validate_array_input(ta, np.float64, 'air temperature')
        
        # Cast initial state as float
        tw_init = float(tw_init)

        # pack input arguments for scipy optimizer
        args = (ta, t_ty, th, tw_init, tw_ice, tw_obs, self._dtype)
        bnds = tuple([self._default_bounds[p] for p in self._param_list])

        # call the actual optimizer function
        res = optimize.differential_evolution(_loss, bounds=bnds, args=args)

        return res

    def pso_fit(self, 
            tw_obs, 
            ta,
            t_ty,
            th = 4.0,
            tw_init = 1.0,
            tw_ice = 0.0,
            swarm_size = 10,
            n_cpus = 1,
            iteration_num = 1000,
            ):
        # Validation check of the inputs
        tw_obs = validate_array_input(tw_obs, np.float64, 'tw_obs')
        ta = validate_array_input(ta, np.float64, 'air temperature')
        
        # Cast initial state as float
        tw_init = float(tw_init)
        # pack input arguments for scipy optimizer
        input_args = (ta, t_ty, th, tw_init, tw_ice, tw_obs, self._dtype, n_cpus)
        constraints = (np.array([self._default_bounds[p][0] for p in self._param_list]),
                       np.array([self._default_bounds[p][1] for p in self._param_list]))
        
        # initialize a optimizer
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=swarm_size, 
                                  dimensions=len(self._param_list), 
                                  options=options, 
                                  bounds=constraints, 
                                  )
        
        cost, joint_vars = optimizer.optimize(_loss_pso, iters = iteration_num, input_args = input_args)
        
        return cost, joint_vars
        
def _loss(X, *args):
    """Return the loss value for the current parameter set.
    """
    # Unpack static arrays
    ta = args[0]
    t_ty = args[1]
    th = args[2]
    tw_init = args[3]
    tw_ice = args[4]
    tw_obs = args[5]
    dtype = args[6]
    
    # Create a custom numpy array of the model parameters
    params = np.zeros(1, dtype=dtype)
    params['a1'] = X[0]
    params['a2'] = X[1]
    params['a3'] = X[2]
    params['a4'] = X[3]
    params['a5'] = X[4]
    params['a6'] = X[5]
    
    # Calculate the simulated lake surface water temperature
    tw = run_air2water6p(ta, t_ty, th, tw_init, tw_ice, params[0])

    # Calculate the Mean-Squared-Error as optimization criterion
    loss_value = calc_mse(tw_obs, tw)

    return loss_value

def _loss_pso(X, input_args):
    """Return the loss value for the current parameter set.
    The shape of X is (n_particles)    
    """
    # Unpack static arrays
    ta = input_args[0]
    t_ty = input_args[1]
    th = input_args[2]
    tw_init = input_args[3]
    tw_ice = input_args[4]
    tw_obs = input_args[5]
    dtype = input_args[6]
    n_cpus = input_args[7]

    # Create a custom numpy array of the model parameters
    # number of particles
    n_particles = X.shape[0]    
    params = np.zeros(n_particles, dtype=dtype)
    params['a1'] = X[:, 0]
    params['a2'] = X[:, 1]
    params['a3'] = X[:, 2]
    params['a4'] = X[:, 3]
    params['a5'] = X[:, 4]
    params['a6'] = X[:, 5]
    
    # run air2water model for particle times get the results
    # could be in parallel
    
    if n_cpus > 1:
        # with Pool(n_cpus) as p:
        #     def run_model(n):
        #         return run_air2water6p(ta, t_ty, th, tw_init, tw_ice, params[n])
        #     tws = p.map(run_model, 
        #                 range(n_particles))  
        foo_ = partial(run_air2water6p, ta, t_ty, th, tw_init, tw_ice)
        tws = Parallel(n_jobs=n_cpus)(delayed(foo_)(i) for i in params)
        
    elif n_cpus == 0:
        tws = [run_air2water6p(ta, t_ty, th, tw_init, tw_ice, params[n]) for n in range(n_particles)]
    else:
        raise ValueError("Choose positive number of the n_cpus")
    
    # Calculate the simulated lake surface water temperature and calculate the mse
    loss_values = np.array([calc_mse(tw_obs, tw) for tw in tws])

    # replace the nan value with 999
    loss_values = np.nan_to_num(loss_values, nan = 999, posinf=1000)

    # print(loss_values)
    return loss_values