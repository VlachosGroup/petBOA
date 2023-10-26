"""
Reactor and numerical integration functions
"""
import numpy as np
from scipy.integrate import ode, solve_ivp
from petboa.utils import RMSE, WeightedRMSE, para_values_to_dict
import timeit

count = 0


# # Assume we have
# # n_rxn reactions and m_spec species
#
# # %% ODE functions for numerical integration


def dcdt(t, concentrations, stoichiometry, rate_expressions, para_dict, names=None, temperature=None, *rate_inputs):
    """
    Compute the derivatives for multiple parallel reactions
    """
    # stoichiometry matrix is in the shape of n_rxn * m_spec
    if not isinstance(stoichiometry[0], list):
        stoichiometry = [stoichiometry]
        n_rxn = 1
    else:
        n_rxn = len(stoichiometry)

    # expand expressions to a list
    if not isinstance(rate_expressions, list):
        rate_expressions = n_rxn * [rate_expressions]
    if not isinstance(names, list):
        names = n_rxn * [names]

    if (n_rxn != len(rate_expressions)) or n_rxn != len(names):
        raise ValueError("Input stoichiometry matrix must equal to the number of input rate expressions or names")

    # rates are in the shape of 1 * n_rxn
    cur_rate = np.zeros(n_rxn)
    for i in range(n_rxn):
        cur_rate[i] = rate_expressions[i](concentrations, para_dict, stoichiometry[i], names[i], temperature,
                                          *rate_inputs)

    # dC/dt for each species
    # dcdt is in the shape of 1 * m_spec 
    # use matrix multiplication
    cur_dcdt = np.matmul(cur_rate, np.array(stoichiometry))

    return cur_dcdt


def ode_solver_ivp(func, y0, t0, tf, t_eval, method, *func_inputs):
    """
    Set up the ode solver 
    Use solve_ivp
    """
    global count
    count += 1
    sol = solve_ivp(func, t_span=[t0, tf], y0=y0, method=method, t_eval=t_eval, args=(*func_inputs,))
    # Extract t and C from sol
    t_vec = sol.t
    C_vec = sol.y
    # ans is a matrix for tC_profile
    # 0th column is the time and ith column is the concentration of species i
    n_species, n_points = sol.y.shape
    ans = np.zeros((n_points, n_species + 1))
    ans[:, 0] = t_vec
    ans[:, 1:] = C_vec.T

    return ans


class Reactor():
    """Reaction ODEs class"""

    def __init__(self, stoichiometry, tf, P0=None, feed_composition=None, C0=None, names=None, temperature=None):
        """Initialize the constants"""
        self.stoichiometry = stoichiometry
        self.names = names  # names of the reactions
        self.P0 = P0
        self.feed_composition = feed_composition
        self.t0 = 0
        self.tf = tf
        self.temperature = temperature

        if (P0 is not None) and (feed_composition is not None):
            self.C0 = P0 * np.array(feed_composition) / np.sum(feed_composition)
        elif C0 is not None:
            self.C0 = C0
        else:
            raise ValueError("Must input P0, feed composition or C0")

    def get_profile(self, rate_expressions, para_dict, t_eval=None, method='LSODA'):
        """Numerical integration of the rate expression given the parameters"""
        # print(method)
        tC_profile = ode_solver_ivp(dcdt, self.C0, self.t0,
                                    self.tf, t_eval, method,
                                    self.stoichiometry,
                                    rate_expressions,
                                    para_dict,
                                    self.names,
                                    self.temperature)

        return tC_profile

    def get_exit_concentration(self, rate_expressions, para_dict, t_eval=None, method='LSODA'):
        """Get the exit concentration of a specie"""

        # Get the profile
        tC_profile = self.get_profile(rate_expressions, para_dict, t_eval, method)

        # Extract the final concentrations by taking out the time 
        Cf = tC_profile[-1, 1:]

        return Cf

    def get_conversion(self, rate_expressions, para_dict, species_indices=0, t_eval=None, method='LSODA'):
        """
        Get the final conversion of a specie
        Species index can be a list or a single integer
        """
        if not isinstance(species_indices, list):
            species_indices = [species_indices]

        # Get the profile
        Cf = self.get_exit_concentration(rate_expressions, para_dict, t_eval, method)

        # Compute the final percentage conversion
        xf = np.zeros(len(species_indices))
        for i, si in enumerate(species_indices):
            xf[i] = (self.C0[si] - Cf[si]) / self.C0[si] * 100

        # Compute the final rates
        dcdt_f = dcdt(self.tf, Cf, self.stoichiometry, rate_expressions, para_dict, self.names, self.temperature)

        return xf, dcdt_f


qoi_choices = ['profile', 'conversion', 'concentration']


class ModelBridge:
    """Parameter estimation class"""

    def __init__(self, rate_expression, para_names, name='estimator_0'):
        """Initialize the rate expression and parameters"""

        self.rate_expression = rate_expression
        self.para_names = para_names
        self.name = name
        self.call_count = 0
        self.model_count = count
        self.param_evolution = []
        self.loss_evolution = []

        # other classes
        self.Reactors = None

    def input_data(
            self,
            stoichiometry,
            reactor_data,
            Y_groundtruth,
            Y_weights=None,
            t_eval=None,
            rxn_names=None,
            qoi='profile',
            species_indices=0,
            method='LSODA'):

        """Initialize n Reactor objects"""

        # stoichiometry matrix is in the shape of n_rxn * m_spec
        if not isinstance(stoichiometry[0], list):
            stoichiometry = [stoichiometry]
            n_rxn = 1
        else:
            n_rxn = len(stoichiometry)

        self.n_rxn = n_rxn
        self.m_specs_total = len(stoichiometry[0])

        # set up reactor objects
        self.Reactors = []
        self.n_reactors = len(reactor_data)

        # evaluate profile parameters
        self.t_eval = t_eval
        self.method = method

        # parse the reactor data and initialize Reactor objects
        for i in range(self.n_reactors):
            reactor_data_i = reactor_data[i]
            Reactor_i = Reactor(stoichiometry, **reactor_data_i)
            if rxn_names is not None:
                Reactor_i.names = rxn_names
            self.Reactors.append(Reactor_i)

        # Set the ground truth or experimental values
        self.Y_groundtruth = Y_groundtruth

        # Set the quantity of interest (QOI)
        if qoi not in qoi_choices:
            raise ValueError("Input qoi must be either profile, conversion or concentration.")
        self.qoi = qoi

        # species index is a list of integers
        # whose conversions that the users care about
        if not isinstance(species_indices, list):
            species_indices = [species_indices]
        self.species_indices = species_indices
        self.m_specs_conversion = len(species_indices)

        # Set the weights for each data point
        if Y_weights is None:
            Y_weights = np.ones((self.n_reactors, 1))

            # if self.qoi == 'profile':
            #    Y_weights = np.ones((self.n_reactors, 1))
            # elif self.qoi == 'conversion':
            #    Y_weights = np.ones((self.n_reactors, self.m_specs))
            # else: # exit concentration
            #    Y_weights = np.ones((self.n_reactors, self.m_specs))

        # normalize the weights
        self.Y_weights = Y_weights / np.sum(Y_weights)

    def conversion(self, xi):
        """Predict the conversions given a set of parameters"""

        para_dict = para_values_to_dict(xi, self.para_names)

        Y_predict = []

        # Compute the first output - conversion
        for i in range(self.n_reactors):
            Reactor_i = self.Reactors[i]
            xf, _ = Reactor_i.get_conversion(self.rate_expression, para_dict,
                                             species_indices=self.species_indices, t_eval=self.t_eval,
                                             method=self.method)
            Y_predict.append(xf)
        self.model_count = count
        return Y_predict

    def exit_concentration(self, xi):
        """Predict the conversions given a set of parameters"""

        para_dict = para_values_to_dict(xi, self.para_names)
        # Y_groundtruth has shape of n_reactors * n_int * m_specs
        Y_predict = []

        # Compute the first output - conversion
        for i in range(self.n_reactors):
            Reactor_i = self.Reactors[i]
            Cf_i = Reactor_i.get_exit_concentration(self.rate_expression, para_dict, t_eval=self.t_eval,
                                                    method=self.method)
            Y_predict.append(Cf_i)
        self.model_count = count
        return Y_predict

    def profile(self, xi, t_eval=None, return_t_eval=True):
        """Predict the conversions given a set of parameters"""
        if t_eval is None:
            t_eval = self.t_eval

        para_dict = para_values_to_dict(xi, self.para_names)
        # Y_groundtruth has shape of n_reactors * n_int * m_specs
        Y_predict = []
        t_predict = []

        # Compute the first output - conversion
        for i in range(self.n_reactors):
            Reactor_i = self.Reactors[i]
            tC_profile_i = Reactor_i.get_profile(self.rate_expression, para_dict, t_eval=t_eval, method=self.method)
            t_predict.append(tC_profile_i[:, 0])
            Y_predict.append(tC_profile_i[:, 1:])  # ignore the first column since it's time
        self.model_count = count
        if return_t_eval:
            return t_predict, Y_predict

        return Y_predict

    def loss_func(self, xi):
        """Generic loss function"""
        loss = 0

        if self.qoi == 'conversion':
            Y_predict = self.conversion(xi)
        elif self.qoi == 'profile':
            Y_predict = self.profile(xi, return_t_eval=False)
        else:  # exit concentration
            Y_predict = self.exit_concentration(xi)

        for i in range(self.n_reactors):
            if self.Y_weights is None:
                loss += RMSE(self.Y_groundtruth[i], Y_predict[i])
            else: #  Factor in the weights
                loss += WeightedRMSE(self.Y_groundtruth[i], Y_predict[i], self.Y_weights)
        self.call_count += 1
        return loss
