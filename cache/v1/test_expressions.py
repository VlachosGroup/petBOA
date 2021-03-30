"""
Tests on expressions module
"""

import numpy as np
from expressions import Reactor, general_rate, temkin_pyzhev_rate, ode_solver, dcdt

# Inputs for the current reaction
stoichiometry = [-1, -3, 2]

# Set temperature
temperature = 623 #K

# Dictionaries for parameters
para_dict_1 = {'K': 1,
                'ksr': 1,
                'KA': 1,
                'KB': 1}

para_dict_2 = {'k1': 1,
                'k2': 1,
                'alpha': 1,
                'beta': 1}

# Test on rates given a set of concentrations
# rate should be 1/27
C_test = np.array([100,300,0])
rate_1 = general_rate(C_test, para_dict_1)
rate_2 = temkin_pyzhev_rate(C_test, para_dict_2)

# Test on the ode solver
P0 = 50 # atm
feed_composition = [1, 3, 0]
C0 = P0 * np.array(feed_composition)/np.sum(feed_composition)
t0 = 0 
tf = 100 # second, from V(=1 cm3) /q (=0.01 cm3/s)

# Numerical integration step
ans_vec = ode_solver(dcdt, C0, t0, tf, stoichiometry, general_rate, para_dict_1)
Cf = ans_vec[-1, 1:] 

# Compute the final percentage conversion
# Use N2 concentrations
xf = (C0[0] - Cf[0])/C0[0] * 100

# Compute the final rates
dcdt_f = dcdt(tf, Cf, stoichiometry, general_rate, para_dict_1)

# Test on the reactor class
reactor_test = Reactor(stoichiometry, temperature, P0, feed_composition, tf)
xf_reactor, _ = reactor_test.get_conversion(general_rate, para_dict_1)