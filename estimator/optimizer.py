"""
Optimizer functions for parameter estimation
"""
import os
import sys
import numpy as np
import copy

from nextorch import plotting, bo, doe, utils, io
from estimator.expressions import general_rate
from estimator.reactor import Reactor
from estimator.utils import WeightedRMSE

import time


class ModelBridge():
    """Parameter estimation class"""
    
    def __init__(self, rate_expression, para_names, name = 'estimator_0'):
        """Initialize the rate experssion and parameters"""
        
        self.rate_expression = rate_expression
        self.para_names = para_names
        self.name = name
        
        # other classes
        self.Reactors = None
        
        
    def input_data(self, stoichiometry, reactor_data, Y_groudtruth, Y_weights=None, t_eval=None, eval_profile=False, method='LSODA'):
        """Initialize n Reactor objects"""
        
        # stoichiometry matrix is in the shape of n_rxn * m_spec
        if not isinstance(stoichiometry[0], list):
            stoichiometry = [stoichiometry]
            n_rxn = 1
        else:
            n_rxn = len(stoichiometry)
        self.n_rxn = n_rxn
        self.m_specs = len(stoichiometry[0])
        
        # set up reactor objects 
        self.Reactors = []
        self.n_reactors = len(reactor_data)

        # evaluate profile parameters
        self.t_eval = t_eval
        self.method = method
        self.eval_profile = eval_profile
    
        # parse the reactor data and initialize Reactor objects
        for i in range(self.n_reactors):
            reactor_data_i = reactor_data[i]
            Reactor_i = Reactor(stoichiometry, **reactor_data_i)
            self.Reactors.append(Reactor_i)
            
        # Set the ground truth or experimental values
        self.Y_groudtruth = Y_groudtruth
        
        # Set the weights for each data point
        if Y_weights is None:
            if eval_profile:
                Y_weights = np.ones((self.n_reactors, 1))
            else:
                Y_weights = np.ones((self.n_reactors, self.m_specs))
        
        self.Y_weights = Y_weights/np.sum(Y_weights)
        
        
    def conversion(self, xi):
        """Predict the conversions given a set of parameters"""
        
        para_dict = {}
        for name_i, para_i in zip(self.para_names, xi):
            para_dict.update({name_i: para_i})    
        
        Y_predict = np.zeros((self.n_reactors, 1))
        
        # Compute the first output - conversion 
        for i in range(self.n_reactors):
            Reactor_i = self.Reactors[i]
            xf, _ = Reactor_i.get_conversion(self.rate_expression, para_dict, t_eval=self.t_eval, method=self.method)
            Y_predict[i] = xf
    
        return Y_predict
    
    
    def profile(self, xi):
        """Predict the conversions given a set of parameters"""
        
        para_dict = {}
        for name_i, para_i in zip(self.para_names, xi):
            para_dict.update({name_i: para_i})    
        
        # Y_groudtruth has shape of n_reactors * n_int * m_specs        
        Y_predict = []
        t_predict = []

        # Compute the first output - conversion 
        for i in range(self.n_reactors):
            Reactor_i = self.Reactors[i]
            tC_profile_i = Reactor_i.get_profile(self.rate_expression, para_dict, t_eval=self.t_eval, method=self.method)
            t_predict.append(tC_profile_i[:, 0]) 
            Y_predict.append(tC_profile_i[:, 1:]) #ignore the first column since it's time
        

        return t_predict, Y_predict
        
    
    def loss_conversion(self, xi):
        """Define the loss function using RMSE"""
        Y_predict = self.conversion(xi)
        
        # Factor in the weights
        loss =  WeightedRMSE(self.Y_groudtruth, Y_predict, self.Y_weights)
        #weighted_diff = (self.Y_groudtruth - Y_predict) 
        #np.sum((weighted_diff)**2 * self.Y_weights)/self.Y_groudtruth.size #MSE
        
        return loss
    
    
    def loss_profile(self, xi):
        """Define the loss function using RMSE"""
        _, Y_predict = self.profile(xi)
        
        loss = 0
        for i in range(self.n_reactors):
            # Factor in the weights
            loss += WeightedRMSE(self.Y_groudtruth[i], Y_predict[i], self.Y_weights)
        
        return loss

    def loss_func(self, xi):
        """Generic loss function"""
        if self.eval_profile:
            return self.loss_profile(xi)
        
        return self.loss_conversion(xi)
        
        


#%% BO Optimizer functions 

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
        #If 1D, make it 2D a matrix
        X_temp = copy.deepcopy(np.array(X))
        if len(X_temp.shape)<2:
            X_temp = np.expand_dims(X_temp, axis=0) #If 1D, make it 2D array
    
        n_points = X_temp.shape[0]
        self.X_temp = X_temp
        xi_list = [] # a list of the columns
        di_fixed = 0 # index for fixed value dimensions
        di_varying = 0 # index for varying x dimensions
    
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
            X_full = np.squeeze(X_full, axis = 0)
        
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
            X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
            
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
    def __init__(self, objective_func, para_ranges):
        super().__init__(objective_func)
        
        self.mask = ParameterMask(para_ranges)
        
        
    def predict(self, X_real):
        """
        vectorized objective function object with fixed and varying inputs
        Input/output matrices
        """
        if len(X_real.shape) < 2:
            X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
            
        Y_real = []
        for i, xi in enumerate(X_real):
            xi_full = self.mask.prepare_X(xi)
            #print(xi_full)
            yi = self.objective_func(xi_full)    
            Y_real.append(yi)
                
        Y_real = np.array(Y_real)
        # Put y in a column
        Y_real = np.expand_dims(Y_real, axis=1)
            
        return Y_real 
        
        
        
class BOOptimizer():
    """
    Automated BO Optimizer
    """
    def __init__(self, name = 'optimizer_0'):
        self.name = name
        
    def optimize(self, 
                 objective_func, 
                 para_ranges, 
                 n_iter = 100, 
                 make_plot = True, 
                 log_flag = False):
        """
        Train a Bayesian Optimizer
        """
        # Vectorize the objective function
        objective_func_vectorized = MaskedFunc(objective_func, para_ranges)
        para_ranges_varying = objective_func_vectorized.mask.para_ranges_varying
        
        # Initialize a BO experiment
        Exp = bo.Experiment(self.name)
        
        # the dimension of inputs
        n_dim = len(para_ranges_varying)
        
        # Latin hypercube design with 10 initial points
        n_init = 5 * n_dim
        X_init = doe.latin_hypercube(n_dim = n_dim, n_points = n_init, seed= 1)
        
        #print(X_init)
        Y_init = bo.eval_objective_func(X_init, para_ranges_varying, objective_func_vectorized.predict)
        
        # Import the initial data
        Exp.input_data(X_init, Y_init, X_ranges=para_ranges_varying, unit_flag=True)
        Exp.set_optim_specs(objective_func=objective_func_vectorized.predict, maximize=False)
        
        # Run optimization loops        
        Exp.run_trials_auto(n_iter)
        
        # Extract the optima
        y_opt, X_opt, index_opt = Exp.get_optim()
        
        # Expand the X into the full size
        X_opt_full = objective_func_vectorized.mask.prepare_X(X_opt)
        
        if make_plot:
            # Plot the optimum discovered in each trial
            plotting.opt_per_trial_exp(Exp, log_flag=log_flag, save_fig=True)

        return X_opt_full, y_opt, Exp
        
        
        
        
        





    
