"""
Optimizer functions for parameter estimation
"""
import os
import sys
import numpy as np
import copy
from nextorch import plotting, bo, doe, utils, io


class ParameterMask():
    """Create full X matrix with fixed values and varying values"""

    def __init__(self, para_ranges, return_1d=True):

        # set the return dimension
        self.return_1d = return_1d

        # the dimension of inputs
        self.n_dim = len(para_ranges)

        self.varying_axes = []
        self.para_ranges_varying = []
        self.para_fixed_values = []

        # Count the number of fixed inputs
        for pi, para_range_i in enumerate(para_ranges):
            if isinstance(para_range_i, list):
                self.para_ranges_varying.append(para_range_i)
                self.varying_axes.append(pi)
            else:
                self.para_fixed_values.append(para_range_i)

    def prepare_X(self, X):
        """Prepare the full X vector"""
        # If 1D, make it 2D a matrix
        X_temp = copy.deepcopy(np.array(X))
        if len(X_temp.shape) < 2:
            X_temp = np.expand_dims(X_temp, axis=0)  # If 1D, make it 2D array

        n_points = X_temp.shape[0]
        self.X_temp = X_temp
        xi_list = []  # a list of the columns
        di_fixed = 0  # index for fixed value dimensions
        di_varying = 0  # index for varying x dimensions

        for di in range(self.n_dim):
            # Get a column from X_test
            if di in self.varying_axes:
                xi = X_temp[:, di_varying]
                di_varying += 1
            # Create a column of fix values
            else:
                fix_value_i = self.para_fixed_values[di_fixed]
                xi = np.ones((n_points, 1)) * fix_value_i
                di_fixed += 1

            xi_list.append(xi)
        # Stack the columns into a matrix
        X_full = np.column_stack(xi_list)

        if self.return_1d:
            X_full = np.squeeze(X_full, axis=0)

        return X_full


class VectorizedFunc():
    """
    Wrapper for the vectorized objective function
    """

    def __init__(self, objective_func):
        self.objective_func = objective_func

    def predict(self, X_real):
        """
        vectorized objective function object
        Input/output matrices
        """
        if len(X_real.shape) < 2:
            X_real = np.expand_dims(X_real, axis=1)  # If 1D, make it 2D array

        Y_real = []
        for i, xi in enumerate(X_real):
            yi = self.objective_func(xi)
            Y_real.append(yi)

        Y_real = np.array(Y_real)
        # Put y in a column
        Y_real = np.expand_dims(Y_real, axis=1)

        return Y_real


class MaskedFunc(VectorizedFunc):
    """
    Wrapper for the objective function with fixed and varying inputs
    """

    def __init__(self, objective_func, para_ranges, kwargs=None):
        super().__init__(objective_func)
        self.kwargs = kwargs
        self.mask = ParameterMask(para_ranges)

    def predict(self, X_real):
        """
        vectorized objective function object with fixed and varying inputs
        Input/output matrices
        """
        if len(X_real.shape) < 2:
            X_real = np.expand_dims(X_real, axis=1)  # If 1D, make it 2D array

        Y_real = []
        for i, xi in enumerate(X_real):
            xi_full = self.mask.prepare_X(xi)
            yi = self.objective_func(xi_full, **self.kwargs)
            Y_real.append(yi)

        Y_real = np.array(Y_real)
        # Put y in a column
        Y_real = np.expand_dims(Y_real, axis=1)

        return Y_real


class BOOptimizer():
    """
    Automated BO Optimizer
    """

    def __init__(self, name='optimizer_0'):
        self.name = name

    def optimize(self,
                 objective_func,
                 para_ranges,
                 acq_func="EI",
                 n_iter=100,
                 n_sample_multiplier=5,
                 make_plot=True,
                 log_flag=False,
                 **kwargs):
        """
        Train a Bayesian Optimizer
        """
        # Vectorize the objective function
        objective_func_vectorized = MaskedFunc(objective_func, para_ranges, kwargs)
        para_ranges_varying = objective_func_vectorized.mask.para_ranges_varying

        # Initialize a BO experiment
        Exp = bo.Experiment(self.name)

        # the dimension of inputs
        n_dim = len(para_ranges_varying)

        # Latin hypercube design with 10 initial points
        n_init = n_sample_multiplier * n_dim
        X_init = doe.latin_hypercube(n_dim=n_dim, n_points=n_init, seed=1)

        Y_init = bo.eval_objective_func(X_init, para_ranges_varying, objective_func_vectorized.predict)

        # Import the initial data
        Exp.input_data(X_init, Y_init, X_ranges=para_ranges_varying, unit_flag=True)
        Exp.set_optim_specs(objective_func=objective_func_vectorized.predict, maximize=False)

        # Run optimization loops        
        Exp.run_trials_auto(n_iter, acq_func_name=acq_func)

        # Extract the optima
        y_opt, X_opt, index_opt = Exp.get_optim()

        # Expand the X into the full size
        X_opt_full = objective_func_vectorized.mask.prepare_X(X_opt)

        if make_plot:
            # Plot the optimum discovered in each trial
            plotting.opt_per_trial_exp(Exp, log_flag=log_flag, save_fig=True)

        return X_opt_full, y_opt, Exp
