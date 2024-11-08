import numbers

import numpy as np

from scipy import optimize

from .basemodel import BaseModel
from .air2water6p_model import run_air2water6p
from ..utils.metrics import calc_mse
from ..utils.array_checks import check_for_negatives, validate_array_input


class leqt(BaseModel):
    """Interface to the lake equilibirum temperature (leqt) model

    This model implements the lake equilibirum temperature (leqt) model
    
    The equation was used from the paper:
    [1] Piccolroaz, S., Zhu, S., Ladwig, R., Carrea, L., Oliver, S., Piotrowski, A. P., ...Zhu, D. Z. (2024). 
    Lake Water Temperature Modeling in an Era of Climate Change: Data Sources, Models, and Future Prospects. 
    Rev. Geophys., 62(1), e2023RG000816. doi: 10.1029/2023RG000816

    Concept is from the paper
    [2] Edinger, J. E., Duttweiler, D. W., & Geyer, J. C. (1968). 
    The Response of Water Temperatures to Meteorological Conditions. 
    Water Resour. Res., 4(5), 1137_1143. doi: 10.1029/WR004i005p01137

    Args:
        params: (optional) Dictonary containing all model parameters as a
            seperate key/value pairs.

    """

    # List of model parameters
    _param_list = ['Keq', 'Ds']

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

    def __init__(self, params=None):
        """Initialize an ABC-Model.

        If no parameters are passed as input arguments, random values are
        sampled that satisfy the parameter constraints of the ABC-Model.

        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.

        """
        super().__init__(params=params)

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


def _loss(X, *args):
    """Return the loss value for the current parameter set."""
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