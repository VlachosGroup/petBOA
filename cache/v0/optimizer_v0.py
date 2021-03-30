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


# Set the global variables (or later assign to self)
# Get the experimental conversion
x_experiment = 21.20
    
# Set the rate expression and parameter names 

# para_names = ['k1', 'k2', 'alpha', 'beta']
# rate_expression = temkin_pyzhev_rate 
# # Set the ranges for each parameter
# X_ranges = [[0, 1], 
#             [0, 1],
#             [0, 10
#             [0, 10]]
para_names = ['K', 'ksr', 'KA', 'KB']
rate_expression = general_rate
# Set the ranges for each parameter
X_ranges = [[0, 1], 
            [0, 1],
            [0, 1],
            [0, 1]]


#%% helper functions
def loss_steady_state(xi):
    """
    Calculate the loss at steady state using the final conversion
    Input should be a vector with a length of 4
    """
    
    
    # Set the constants
    P0 = 50 # atm
    feed_composition = [1, 3, 0]
    stoichiometry = [-1, -3, 2]
    tf = 100 # second, from V(=1 cm3) /q (=0.01 cm3/s)
    
    # Initialize a reactor 
    reactor = Reactor(stoichiometry, P0, feed_composition, tf)
    
    para_dict = {}
    for name_i, para_i in zip(para_names, xi):
        para_dict.update({name_i: para_i})    
    
    # Compute the loss, only keep the first output -conversion  
    xf, _ = reactor.get_conversion(rate_expression, para_dict)
    loss = np.abs(x_experiment - xf)
    
    return loss


def objective_func(X_real):
    """
    Objective function object
    Input/output matrices
    """
    if len(X_real.shape) < 2:
        X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        
    Y_real = []
    for i, xi in enumerate(X_real):
        yi = loss_steady_state(xi)    
        Y_real.append(yi)
            
    Y_real = np.array(Y_real)
    # Put y in a column
    Y_real = np.expand_dims(Y_real, axis=1)
        
    return Y_real



def optimizater(X_ranges, n_iter, make_plot = True):
    """Train a Bayesian Optimizer
    """
    n_dim = len(X_ranges) # the dimension of inputs

    Exp = bo.Experiment('parameter_estimation')
    
    # Latin hypercube design with 10 initial points
    n_init = 5 * n_dim
    X_init = doe.latin_hypercube(n_dim = n_dim, n_points = n_init, seed= 1)
    Y_init = bo.eval_objective_func(X_init, X_ranges, objective_func)
    # Import the initial data
    Exp.input_data(X_init, Y_init, X_ranges = X_ranges)
    Exp.set_optim_specs(objective_func = objective_func, maximize =  False)
    
    # Run optimization loops        
    Exp.run_trials_auto(n_iter)
    
    # Extract the optima
    y_opt, X_opt, index_opt = Exp.get_optim()
    
    if make_plot:
        # Plot the optimum discovered in each trial
        plotting.opt_per_trial_exp(Exp)

    return y_opt, X_opt, Exp

#%% Tests 
n_iter = 30
# start a timer
start_time = time.time()
y_opt, X_opt, Exp = optimizater(X_ranges, n_iter)
end_time= time.time()
# Print the results
print('Paramter estimation takes {:.2f} min'.format((end_time-start_time)/60))
print('Final loss {:.3f}'.format(y_opt))
print('Parameters are {}'.format(X_opt))