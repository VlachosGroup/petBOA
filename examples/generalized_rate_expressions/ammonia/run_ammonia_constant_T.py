"""
The main testing file for
estimating parameters for ammonia chemistry
Use N2 conversion as a metric for gauging responses
"""
import time

import numpy as np
import pandas as pd
from nextorch import plotting

import estimator.utils as ut
from estimator.expressions import general_rate
from estimator.optimizer import BOOptimizer
from estimator.reactor import ModelBridge

# %% Import and process data
# Set the reaction constants
stoichiometry = [-1, -3, 2]

data = pd.read_excel('ammonia_data.xlsx')
T = 623.0  # K

# Use the data at a single T only
data_single_T = data[np.array(data['Temperature (K)'], dtype=float) == T]
n_experiments = len(data_single_T)

# Select the pressure
pressures = np.array(data_single_T['Pressure (atm)'], dtype=float)

# Select the conversions
conversions = np.array(data_single_T['Conversion'], dtype=float)
conversions = np.expand_dims(conversions, axis=1)
Y_experiments = conversions

# Select the feed compositions
feed_ratios = np.array(data_single_T['Feed ratio (H2:N2)'], dtype=float)

# Select the residence time
times = np.array(data_single_T['Residence time (s)'], dtype=float)

# Select the weights
Y_weights = np.array(data_single_T['Weight'], dtype=float)

# Construct a list of dictionaries as Reactor inputs
experimental_data = []

for i in range(n_experiments):
    experiment_i = {'P0': pressures[i], 'tf': times[i]}
    feed_composition_i = np.zeros(3)
    feed_composition_i[0] = 1
    feed_composition_i[1] = feed_ratios[i]
    experiment_i['feed_composition'] = feed_composition_i
    experimental_data.append(experiment_i)

# %% Parameter estimation section
# estimator name 
estimator_name = 'ammonia_constant_T_results'

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
bridge = ModelBridge(rate_expression, para_names_1, name=estimator_name)
bridge.input_data(stoichiometry, experimental_data, Y_experiments, Y_weights, qoi='conversion')

# set up an optimizer
optimizer = BOOptimizer(estimator_name)
X_opt, loss_opt, Exp = optimizer.optimize(bridge.loss_func, para_ranges_1, n_iter)
end_time = time.time()

# Predict the conversions given the optimal set of parameters
Y_opt = np.array(bridge.conversion(X_opt))
plotting.parity(Y_opt, Y_experiments)

# Print the results
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)
