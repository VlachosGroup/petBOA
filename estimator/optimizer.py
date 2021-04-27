"""
Optimizer functions for parameter estimation
"""
import os
import sys
import numpy as np


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
        
        
    def conversion_steady_state(self, xi):
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
        
    
    def loss_steady_state(self, xi):
        """Define the loss function using RMSE"""
        Y_predict = self.conversion_steady_state(xi)
        
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

        if self.eval_profile:
            return self.loss_profile(xi)
        
        return self.loss_steady_state(xi)
        
        


#%% BO Optimizer functions    
class VectorizedFunc():
    """
    Wrapper for the objective function
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
        objective_func_vectorized = VectorizedFunc(objective_func)
        
        # Initialize a BO experiment
        Exp = bo.Experiment(self.name)
        
        # the dimension of inputs
        n_dim = len(para_ranges)
        
        # Latin hypercube design with 10 initial points
        n_init = 5 * n_dim
        X_init = doe.latin_hypercube(n_dim = n_dim, n_points = n_init, seed= 1)
        Y_init = bo.eval_objective_func(X_init, para_ranges, objective_func_vectorized.predict)
        
        # Import the initial data
        Exp.input_data(X_init, Y_init, X_ranges=para_ranges, unit_flag=True)
        Exp.set_optim_specs(objective_func=objective_func_vectorized.predict, maximize=False)
        
        # Run optimization loops        
        Exp.run_trials_auto(n_iter)
        
        # Extract the optima
        y_opt, X_opt, index_opt = Exp.get_optim()
        
        if make_plot:
            # Plot the optimum discovered in each trial
            plotting.opt_per_trial_exp(Exp, log_flag=log_flag)

        return X_opt, y_opt, Exp
        
        
        
        
        





    
