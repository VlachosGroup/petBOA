"""
The main testing file for BO and expressions
"""
import pandas as pd
import numpy as np
import time

from optimizer import Estimator
from expressions import Reactor, general_rate, temkin_pyzhev_rate
from nextorch import plotting, bo, doe, utils, io

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
    
    reactor_i['temperature' ] = T
    reactor_i['P0'] = pressures[i]
    reactor_i['tf'] = times[i]
    
    feed_composition_i = np.zeros(3)
    feed_composition_i[0] = 1
    feed_composition_i[1] = feed_ratios[i]
    reactor_i['feed_composition'] = feed_composition_i
    
    reactor_data.append(reactor_i)


#%% Parameter estimation section
# Set the number of optimization loops
n_iter = 30

# Set the rate expression and parameter names 
para_names_1 = ['K', 'ksr', 'KA', 'KB']
rate_expression_1 = general_rate

# Set the ranges for each parameter
para_ranges_1 = [[0, 0.5], 
                [0, 0.5],
                [0, 0.5],
                [0, 0.5]]

# start a timer
start_time = time.time()
estimator_1 = Estimator(rate_expression_1, para_names_1, para_ranges_1, name = 'rate_1')
estimator_1.input_data(stoichiometry, reactor_data, Y_experiments, Y_weights)
X_opt_1, Y_opt_1, loss_opt_1, Exp_1 = estimator_1.optimize(n_iter)
end_time= time.time()

# Print the results
file_1 = open(estimator_1.name + ".txt","w")
file_1.write('Parameter estimation takes {:.2f} min \n'.format((end_time-start_time)/60))
file_1.write('Final loss {:.3f} \n'.format(loss_opt_1))
file_1.write('Parameters are {} \n'.format(X_opt_1))
file_1.close()

#%% Second rate expression and parameter names
para_names_2 = ['k1', 'k2', 'alpha', 'beta']
rate_expression_2 = temkin_pyzhev_rate
# Set the ranges for each parameter
para_ranges_2 = [[0, 1],
                [0, 1],
                [0, 10],
                [0, 10]]
# start a timer
start_time = time.time()
estimator_2 = Estimator(rate_expression_2, para_names_2, para_ranges_2, name = 'rate_2')
estimator_2.input_data(stoichiometry, reactor_data, Y_experiments, Y_weights)
X_opt_2, Y_opt_2, loss_opt_2, Exp_2 = estimator_2.optimize(n_iter)
end_time = time.time()

# Print the results
file_2 = open(estimator_2.name + ".txt","w")
file_2.write('Parameter estimation takes {:.2f} min \n'.format((end_time-start_time)/60))
file_2.write('Final loss {:.3f} \n'.format(loss_opt_2))
file_2.write('Parameters are {} \n'.format(X_opt_2))
file_2.close()


# Compare two models
plotting.opt_per_trial([Exp_1.Y_real, Exp_2.Y_real], 
                        maximize=False, 
                        design_names = ['General', 'Temkin Pyzhev'])

# Temkin fits badly, any idea what's the reasonable of parameters?