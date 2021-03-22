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
    
    def __init__(self, rate_expression, para_names, para_ranges, name = 'parameter_estimation'):
        """Initialize the rate experssion and parameters"""
        
        self.rate_expression = rate_expression
        self.para_names = para_names
        self.para_ranges = para_ranges
        self.name = name
        # the dimension of inputs
        self.n_dim = len(para_names) 
        # other classes
        self.Reactor = None
        self.BOExperiment = None
        
    def input_reactor(self, stoichiometry, P0, feed_composition, tf, x_groudtruth):
        """Initialize a reactor"""
        self.Reactor = Reactor(stoichiometry, P0, feed_composition, tf)
        # Set the ground truth or experimental values
        self.x_groudtruth = x_groudtruth
        
    def loss_steady_state(self, xi):
        """Define the loss function"""
        para_dict = {}
        for name_i, para_i in zip(self.para_names, xi):
            para_dict.update({name_i: para_i})    
        
        # Compute the loss, only keep the first output -conversion  
        xf, _ = self.Reactor.get_conversion(self.rate_expression, para_dict)
        loss = np.abs(self.x_groudtruth - xf)
        
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
        y_opt, X_opt, index_opt = Exp.get_optim()
        
        if make_plot:
            # Plot the optimum discovered in each trial
            plotting.opt_per_trial_exp(Exp)
        
        # Assign the experiment to self
        self.BOExperiment = Exp

        return y_opt, X_opt, Exp


#%% Tests 

# Set the reaction constants
P0 = 50 # atm
feed_composition = [1, 3, 0]
stoichiometry = [-1, -3, 2]
tf = 100 # second, from V(=1 cm3) /q (=0.01 cm3/s)
    
# Get the experimental conversion
x_experiment = 21.20


# Set the number of optimization loops
n_iter = 30


# Set the rate expression and parameter names 
para_names_1 = ['K', 'ksr', 'KA', 'KB']
rate_expression_1 = general_rate

# Set the ranges for each parameter
para_ranges_1 = [[0, 1], 
                [0, 1],
                [0, 1],
                [0, 1]]

# start a timer
start_time = time.time()
estimator_1 = Estimator(rate_expression_1, para_names_1, para_ranges_1)
estimator_1.input_reactor(stoichiometry, P0, feed_composition, tf, x_experiment)
y_opt_1, X_opt_1, Exp_1 = estimator_1.optimize(n_iter)
end_time= time.time()

# Print the results
print('Paramter estimation takes {:.2f} min'.format((end_time-start_time)/60))
print('Final loss {:.3f}'.format(y_opt_1))
print('Parameters are {}'.format(X_opt_1))


# Second rate expression and parameter names
para_names_2 = ['k1', 'k2', 'alpha', 'beta']
rate_expression_2 = temkin_pyzhev_rate
# Set the ranges for each parameter
para_ranges_2 = [[0, 1],
                [0, 1],
                [0, 10],
                [0, 10]]
# start a timer
start_time = time.time()
estimator_2 = Estimator(rate_expression_2, para_names_2, para_ranges_2)
estimator_2.input_reactor(stoichiometry, P0, feed_composition, tf, x_experiment)
y_opt_2, X_opt_2, Exp_2 = estimator_2.optimize(n_iter)
end_time= time.time()

# Print the results
print('Paramter estimation takes {:.2f} min'.format((end_time-start_time)/60))
print('Final loss {:.3f}'.format(y_opt_2))
print('Parameters are {}'.format(X_opt_2))


# Compare two models
plotting.opt_per_trial([Exp_1.Y_real, Exp_2.Y_real], 
                       maximize=False, 
                       design_names = ['General', 'Temkin Pyzhev'])