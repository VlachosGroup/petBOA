"""
Optimizer functions for parameter estimation
"""
import os
import sys
import numpy as np

nextorch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../nextorch/'))
sys.path.insert(0, nextorch_path)

from nextorch import plotting, bo, doe, utils, io


def compute_rates(ode_solver, dcdt, C0, t0, tf, rate_expression, stoichiometry, para_dict):
    
    ans = ode_solver(dcdt, C0, t0, tf, rate_expression, stoichiometry, para_dict)
    return ans
    
    

def loss_transient(ode_solver, dcdt, C0, t0, tf, rate_expression, stoichiometry, para_dict, experimental_rates):
    # for transient, use spine fit 
    
    rates = compute_rates(ode_solver, dcdt, C0, t0, tf, rate_expression, stoichiometry, para_dict)
    loss = np.linalg.norm(experimental_rates, rates)
    
    return loss



def loss_steady_state():
    # for steady state, use the final output 
    pass


def objective_func(X_real):
    if len(X_real.shape) < 2:
        X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        
    Y_real = []
    for i, xi in enumerate(X_real):
        # Need to change here
        # Conditions = {'T_degC (C)': xi[0], 'pH': xi[1], 'tf (min)' : 10**xi[2]}
        # yi, _ = Reactor(**Conditions) # only keep the first output       
        Y_real.append(yi)
            
    Y_real = np.array(Y_real)
    # Put y in a column
    Y_real = np.expand_dims(Y_real, axis=1)
        
    return Y_real # yield




def optimizater(X_names, X_ranges, n_iter):
    """Train a Bayesian Optimizer
    """
    n_dim = len(X_names) # the dimension of inputs

    Exp = bo.Experiment('parameter_estimation')
    
    # Latin hypercube design with 10 initial points
    n_init = 10
    X_init = doe.latin_hypercube(n_dim = n_dim, n_points = n_init, seed= 1)
    Y_init = bo.eval_objective_func(X_init, X_ranges, objective_func)
    # Import the initial data
    Exp.input(X_init, Y_init, X_ranges = X_ranges)
    Exp.set_optim_specs(objective_func = objective_func, maximize =  False)
    
    
    for i in range(n_iter):
        Exp.run_trials_auto()
        
    y_opt, X_opt, index_opt = Exp.get_optim()

    return X_opt