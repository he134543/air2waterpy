import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
from joblib import Parallel, delayed
from functools import partial
from .air2water_model import run_air2water
from .metrics import calc_mse, calc_nse, calc_r2
from .gen_params import get_param_bound, find_wider_range


class air2water():
    """Interface to the the air2water model.
    This model implements the 6/8-parameter version of air2water model
    Original Publication:
        Piccolroaz S., M. Toffolon, and B. Majone (2013), 
        A simple lumped model to convert air temperature into surface water temperature in lakes, 
        Hydrol. Earth Syst. Sci., 17, 3323-3338, doi:10.5194/hess-17-3323-2013
    """
    def __init__(self,
                 params=None,
                 version="6p"):
        """Initialize an air2water model

        Args:
            params: (optional) Dictonary containing all model parameters as a seperate key/value pairs. Default to None
            version: (optional) String indicating which version is the air2water model. support "6p" and "8p". Default to "8p".
        """
        # load model version
        self._model_version = version
        if version == "6p":
            # List of model parameters
            self._param_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

            # Initiate a default boundary based on physics meaning
            self._default_bounds = {'a1': (1e-4, 2),
                            'a2': (1e-4, 0.5),
                            'a3': (1e-4, 0.5),
                            'a4': (1, 50),
                            'a5': (1e-4, 1.2),
                            'a6': (0, 1)}
            # Custom numpy datatype needed for numba input
            self._dtype = np.dtype([('a1', np.float64),
                            ('a2', np.float64),
                            ('a3', np.float64),
                            ('a4', np.float64),
                            ('a5', np.float64),
                            ('a6', np.float64)])
        
        elif version == "8p":
            # List of model parameters
            self._param_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

            # Initiate a default boundary based on physics meaning
            self._default_bounds = {'a1': (1e-4, 2),
                            'a2': (1e-4, 0.5),
                            'a3': (1e-4, 0.5),
                            'a4': (1, 50),
                            'a5': (1e-4, 1.2),
                            'a6': (0, 1),
                            'a7': (0, 150),
                            'a8': (0, 0.5)}
            # Custom numpy datatype needed for numba input
            self._dtype = np.dtype([('a1', np.float64),
                            ('a2', np.float64),
                            ('a3', np.float64),
                            ('a4', np.float64),
                            ('a5', np.float64),
                            ('a6', np.float64),
                            ('a7', np.float64),
                            ('a8', np.float64)])
        else:
            raise ValueError("Select the correct model version. Choose either '6p' or '8p'.")
        
        # Randomly generate a parameter if no params was passed
        if params == None:
            num = 1
            params = np.zeros(num, dtype=self._dtype)
            # sample one value for each parameter
            for param in self._param_list:
                values = np.random.uniform(low=self._default_bounds[param][0],
                                high=self._default_bounds[param][1],
                                size=num)
                params[param] = values
            self.params = params
        elif isinstance(params, dict):
            self.load_param(params)
        else:
            raise ValueError("Set params as None or use python dictionary")
        
    def load_params(self, params):
        """A function to load parameters into the model

        Args:
            params (Dict): a python dictionary contain model parameters
        """
        num = 1
        params_dict = params.copy()
        params = np.zeros(num, dtype=self._dtype)
        for key, value in params_dict.items():
            params[key] = value
        # load parameters
        self.params = params
            
        
    def update_param_bnds(self, 
                        mean_depth_range,
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
        """A function to estimate the potential parameter range based on the lake characteristics. 
        This function is adapted from the matlab (https://github.com/spiccolroaz/air2water/) and also the R script (https://github.com/aemon-j/air2wateR)
        Instead of directly give a mean lake depth, this function allows providing a range of potential lake mean depth.

        Args:
            mean_depth_range (_tuple_): _Possible range of the estimated depth_
            tw_range (tuple, optional): _Range of all possible temperature_. Defaults to (0, 30).
            sradmax_range (tuple, optional): _Range of daily maximum shortwave solar radiation_. Defaults to (200, 450).
            sradmin_range (tuple, optional): _Range of daily minimum shortwave solar radiation_. Defaults to (0, 200).
            albedo_range (tuple, optional): _Range of daily minimum shortwave solar radiation_. Defaults to (0.04, 0.2).
            epsilon_a_range (tuple, optional): _Range of epsilon a_. Defaults to (0.6, 0.9).
            alpha_s_range (tuple, optional): _Range of albedo_. Defaults to (3,15).
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
        if self._model_version == "8p":
            for i in range(8):
                self._default_bounds[self._param_list[i]] = find_wider_range(bnds_deep[i], bnds_shallow[i])
        elif self._model_version == "8p":
            for i in range(6):
                self._default_bounds[self._param_list[i]] = find_wider_range(bnds_deep[i], bnds_shallow[i])
        else:
            raise ValueError("Select model version '6p' or '8p'")

    def simulate(self,
                 ta,
                 period,
                 th = 4.0,
                 tw_init = 1.0,
                 tw_ice = 0.0, 
                 ):
        """Simulate the lake surface water temperature for the passed air temperature. This function is adapted from the RRMPG: https://github.com/kratzert/RRMPG.git
        
        Args:
            ta (pandas.Series): air temperature data for each timestep. Preferably numpy array, could be a list or pandas series.
            period (pandas.Series): simulation period, use pd.Timestamp to indicate time. Preferably pandas series.
            th: deep water temperature. Const
            tw_init: (optional) Initial value for the lake surface water temperature.
        """
        # calculate the proportion of currrent day of year to the total number of days on the year
        doy = np.array([dt.dayofyear for dt in period])
        tdoy = np.array([pd.Timestamp(dt.year, 12, 31).dayofyear for dt in period])
        t_ty = doy/tdoy
        
        # turn ta to numpy array
        if isinstance(ta, pd.Series):
            ta = ta.to_numpy().ravel()
        elif isinstance(ta, np.array):
            ta = ta.ravel()
        elif isinstance(ta, list):
            ta = np.array(ta).ravel()
        else:
            raise ValueError("Air temp should be one of the numpy array, pandas series and python list")
        
        # use model params
        params = self.params[0]
        
        # call simulation function given the parameter
        tw = run_air2water(ta, t_ty, th, tw_init, tw_ice, self._model_version, params)

        # build pandas dataframe
        tw = pd.DataFrame(tw, index = period, columns = ["tw_sim"])
        
        return tw

    def pso_fit(self,
            tw_obs,
            ta,
            period,
            obj_func = "MSE",
            th = 4.0,
            tw_init = 1.0,
            tw_ice = 0.0,
            swarm_size = 100,
            n_cpus = 1,
            iteration_num = 500,
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            ):
        """This function uses Particle Swarm Optimization algorithm to search for the best parameter combination.

        Args:
            tw_obs (np.array): A numpy array of lake surface water temperature observations. Missing value should be set as np.NaN
            ta (np.array): A numpy array of air temperature. Should be continous.
            period (pandas.Series): simulation period, use pd.Timestamp to indicate time. Preferably pandas series.
            obj_func (str, optional): Objective function for calibration. Could be `MSE`, `NSE`. Defaults to `MSE`.
            th (float, optional): _description_. Defaults to 4.0.
            tw_init (float, optional): _description_. Defaults to 1.0.
            tw_ice (float, optional): _description_. Defaults to 0.0.
            swarm_size (int, optional): _description_. Defaults to 100.
            n_cpus (int, optional): _description_. Defaults to 1.
            iteration_num (int, optional): _description_. Defaults to 500.
            options (dict, optional): _description_. Defaults to {'c1': 0.5, 'c2': 0.3, 'w': 0.9}.

        Returns:
            loss and parameters
        """
        
        # calculate the proportion of currrent day of year to the total number of days on the year
        doy = np.array([dt.dayofyear for dt in period])
        tdoy = np.array([pd.Timestamp(dt.year, 12, 31).dayofyear for dt in period])
        t_ty = doy/tdoy
        
        # turn ta to numpy array
        if isinstance(ta, pd.Series):
            ta = ta.to_numpy().ravel()
        elif isinstance(ta, np.array):
            ta = ta.ravel()
        elif isinstance(ta, list):
            ta = np.array(ta).ravel()
        else:
            raise ValueError("Air temp should be one of the numpy array, pandas series and python list")
        
        # Cast initial state as float
        tw_init = float(tw_init)
        # pack input arguments for scipy optimizer
        input_args = (ta, t_ty, th, tw_init, tw_ice, tw_obs, self._dtype, self._model_version, n_cpus, obj_func)
        constraints = (np.array([self._default_bounds[p][0] for p in self._param_list]),
                       np.array([self._default_bounds[p][1] for p in self._param_list]))
        
        # initialize a optimizer
        optimizer = GlobalBestPSO(n_particles=swarm_size, 
                                  dimensions=len(self._param_list), 
                                  options=options, 
                                  bounds=constraints, 
                                  )
        
        cost, joint_vars = optimizer.optimize(_loss_pso, iters = iteration_num, input_args = input_args)
        
        return cost, joint_vars
        
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
    model_version = input_args[7]
    n_cpus = input_args[8]
    obj_func = input_args[9]
    
    # Create a custom numpy array of the model parameters
    # number of particles
    n_particles = X.shape[0]    
    params = np.zeros(n_particles, dtype=dtype)    
    if model_version == "6p":
        params['a1'] = X[:, 0]
        params['a2'] = X[:, 1]
        params['a3'] = X[:, 2]
        params['a4'] = X[:, 3]
        params['a5'] = X[:, 4]
        params['a6'] = X[:, 5]
    elif model_version == "8p":
        params['a1'] = X[:, 0]
        params['a2'] = X[:, 1]
        params['a3'] = X[:, 2]
        params['a4'] = X[:, 3]
        params['a5'] = X[:, 4]
        params['a6'] = X[:, 5]
        params['a7'] = X[:, 6]
        params['a8'] = X[:, 7]
    
    # run air2water model for particle times get the results
    if n_cpus > 1:
        # parallel
        foo_ = partial(run_air2water, ta, t_ty, th, tw_init, tw_ice, model_version)
        tws = Parallel(n_jobs=n_cpus)(delayed(foo_)(i) for i in params)
    elif n_cpus == 1:
        tws = [run_air2water(ta, t_ty, th, tw_init, tw_ice, model_version, params[n]) for n in range(n_particles)]
    else:
        raise ValueError("Choose a positive number of the n_cpus")
    
    if obj_func == "MSE":    
        # Calculate the simulated lake surface water temperature and calculate the mse
        loss_values = np.array([calc_mse(tw_obs, tw) for tw in tws])
    elif obj_func == "NSE":
        # Note the NSE here will be -NSE
        loss_values = np.array([-calc_nse(tw_obs, tw) for tw in tws])
    elif obj_func == "R2":
        loss_values = np.array([-calc_r2(tw_obs, tw) for tw in tws])
    else:
        raise ValueError(f"Current objective functions do not support {obj_func}. Consider use MSE, NSE or R2")
    # # replace the nan value with 999
    # loss_values = np.nan_to_num(loss_values, nan = 999, posinf=1000)
    # print(loss_values)
    return loss_values