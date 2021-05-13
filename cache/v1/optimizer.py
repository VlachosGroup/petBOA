"""
Optimizer functions for parameter estimation
"""
import os
import sys
import numpy as np

nextorch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../nextorch/'))
sys.path.insert(0, nextorch_path)

from nextorch import plotting, bo, doe, utils, io
from expressions import Reactor, general_rate, temkin_pyzhev_rate
import time


class Estimator():
    """Parameter estimation class"""
    
    def __init__(self, rate_expression, para_names, para_ranges, name = 'estimator_0'):
        """Initialize the rate experssion and parameters"""
        
        self.rate_expression = rate_expression
        self.para_names = para_names
        self.para_ranges = para_ranges
        self.name = name
        # the dimension of inputs
        self.n_dim = len(para_names) 
        # other classes
        self.Reactors = None
        self.BOExperiment = None
        
    def input_data(self, stoichiometry, reactor_data, Y_groudtruth, Y_weights=None):
        """Initialize n Reactor objects"""
        
        self.Reactors = []
        self.n_reactors = len(reactor_data)
        
        # parse the reactor data and initialize Reactor objects
        for i in range(self.n_reactors):
            reactor_data_i = reactor_data[i]
            Reactor_i = Reactor(stoichiometry, **reactor_data_i)
            self.Reactors.append(Reactor_i)
            
        # Set the ground truth or experimental values
        self.Y_groudtruth = Y_groudtruth
        
        # Set the weights for each data point
        if Y_weights is None:
            Y_weights = np.ones((self.n_reactors, 1))
        self.Y_weights = Y_weights/np.sum(Y_weights)
        
        
    def predict(self, xi):
        """Predict the conversions given a set of parameters"""
        
        para_dict = {}
        for name_i, para_i in zip(self.para_names, xi):
            para_dict.update({name_i: para_i})    
        
        Y_predict = np.zeros((self.n_reactors, 1))
        
        # Compute the first output - conversion 
        for i in range(self.n_reactors):
            Reactor_i = self.Reactors[i]
            xf, _ = Reactor_i.get_conversion(self.rate_expression, para_dict)
            Y_predict[i] = xf
            
        return Y_predict
        
    def loss_steady_state(self, xi):
        """Define the loss function using RMSE"""
        Y_predict = self.predict(xi)
        
        # Factor in the weights
        weighted_diff = (self.Y_groudtruth - Y_predict) * self.Y_weights 
        loss = np.linalg.norm(weighted_diff)**2/self.n_reactors
        
        return loss
    
    def objective_func(self, X_real):
        """
        Objective function object
        Input/output matrices
        """
        if len(X_real.shape) < 2:
            X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
            
        Y_real = []
        for i, xi in enumerate(X_real):
            yi = self.loss_steady_state(xi)    
            Y_real.append(yi)
                
        Y_real = np.array(Y_real)
        # Put y in a column
        Y_real = np.expand_dims(Y_real, axis=1)
            
        return Y_real 
    
    def optimize(self, n_iter, make_plot = True):
        """Train a Bayesian Optimizer
        """
        # Initialize a BO experiment
        Exp = bo.Experiment(self.name)
        
        # Latin hypercube design with 10 initial points
        n_init = 5 * self.n_dim
        X_init = doe.latin_hypercube(n_dim = self.n_dim, n_points = n_init, seed= 1)
        Y_init = bo.eval_objective_func(X_init, self.para_ranges, self.objective_func)
        # Import the initial data
        Exp.input_data(X_init, Y_init, X_ranges = self.para_ranges)
        Exp.set_optim_specs(objective_func = self.objective_func, maximize =  False)
        
        # Run optimization loops        
        Exp.run_trials_auto(n_iter)
        
        # Extract the optima
        loss_opt, X_opt, index_opt = Exp.get_optim()
        
        # Predict the Y at optima
        Y_opt = self.predict(X_opt)
        
        if make_plot:
            # Plot the optimum discovered in each trial
            plotting.opt_per_trial_exp(Exp)
            plotting.parity(self.Y_groudtruth, Y_opt)
        
        # Assign the experiment to self
        self.BOExperiment = Exp

        return X_opt, Y_opt, loss_opt, Exp





    