"""
The main testing file for BO and expressions
"""
import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_path)

import pandas as pd
import numpy as np
import time

from nextorch import plotting
from estimator.optimizer import ModelBridge, BOOptimizer
from estimator.expressions import  general_rate
from estimator.reactor import Reactor

#%% Import and process data
# Set the reaction constants
stoichiometry = [-1, -3, 2]

data = pd.read_excel('ammonia_data.xlsx')
T = 623.0 # K

# Use the data at a single T only
data_single_T = data[np.array(data['Temperature (K)'], dtype = float) == T]
n_reactors = len(data_single_T)

# Select the pressure
pressures = np.array(data_single_T['Pressure (atm)'], dtype = float)

# Select the conversions
conversions = np.array(data_single_T['Conversion'], dtype = float)
conversions = np.expand_dims(conversions, axis = 1)
Y_experiments = conversions

# Selet the feed compositions
feed_ratios = np.array(data_single_T['Feed ratio (H2:N2)'], dtype = float)

# Select the residence time
times = np.array(data_single_T['Residence time (s)'], dtype = float)

# Select the weights
Y_weights = np.array(data_single_T['Weight'], dtype = float)

# Construct a list of dictionaries as Reactor inputs
reactor_data = []

for i in range(n_reactors):
    
    reactor_i = {}
    
    #reactor_i['temperature' ] = T
    reactor_i['P0'] = pressures[i]
    reactor_i['tf'] = times[i]
    
    feed_composition_i = np.zeros(3)
    feed_composition_i[0] = 1
    feed_composition_i[1] = feed_ratios[i]
    reactor_i['feed_composition'] = feed_composition_i
    
    reactor_data.append(reactor_i)


#%% Parameter estimation section
# Set the number of optimization loops
n_iter = 50

# Set the rate expression and parameter names 
para_names_1 = ['K', 'ksr', 'KA', 'KB']
rate_expression = general_rate

# Set the ranges for each parameter
para_ranges_1 = [[0.01, 0.5], 
                [0.01, 0.5],
                [0.01, 0.5],
                [0.01, 0.5]]


# start a timer
start_time = time.time()

# Input experimental data and models (rate expressions) into a model bridge
bridge = ModelBridge(rate_expression, para_names_1, name = 'rate_1')
bridge.input_data(stoichiometry, reactor_data, Y_experiments, Y_weights)

# set up an optimzer 
optimizer = BOOptimizer()
X_opt, loss_opt, Exp = optimizer.optimize(bridge.loss_func, para_ranges_1, n_iter)
end_time= time.time()

# Predict the conversions given the optimal set of paraemeters
Y_opt = bridge.conversion_steady_state(X_opt)
plotting.parity(Y_opt, Y_experiments)

# Print the results
file = open(bridge.name + ".txt","w")
file.write('Parameter estimation takes {:.2f} min \n'.format((end_time-start_time)/60))
file.write('Final loss {:.3f} \n'.format(loss_opt))
file.write('Parameters are {} \n'.format(X_opt))
file.close()

