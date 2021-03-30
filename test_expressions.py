"""
Tests on expressions module
"""

import numpy as np
from expressions import Reactor, general_rate, ode_solver, dcdt


# Inputs for the current reaction
stoichiometry = [-1, -3, 2]

# Test on the ode solver
P0 = 50 # atm
feed_composition = [1, 3, 0]
C0 = P0 * np.array(feed_composition)/np.sum(feed_composition)
t0 = 0 
tf = 100 # second, from V(=1 cm3) /q (=0.01 cm3/s)

C_test = np.array([100,300,0])

#%% Tests with no temperature dependence
# Dictionaries for parameters
para_dict_1 = {'K': 1,
                'ksr': 1,
                'KA': 1,
                'KB': 1}

# Test on rates given a set of concentrations
# rate should be 1/27
rate_1 = general_rate(C_test, para_dict_1)


# Numerical integration step
ans_vec = ode_solver(dcdt, C0, t0, tf, stoichiometry, general_rate, para_dict_1)
Cf = ans_vec[-1, 1:] 

# Compute the final percentage conversion
# Use N2 concentrations
xf_1 = (C0[0] - Cf[0])/C0[0] * 100

# Compute the final rates
dcdt_f_1 = dcdt(tf, Cf, stoichiometry, general_rate, para_dict_1)

# Test on the reactor class
# xf_reactor should == xf
reactor_test_1 = Reactor(stoichiometry, P0, feed_composition, tf)
xf_reactor_1, _ = reactor_test_1.get_conversion(general_rate, para_dict_1)


#%% Tests with temperature dependence
para_dict_2 = {'K_prefactor': 1,
                'ksr_prefactor': 1,
                'KA_prefactor': 1,
                'KB_prefactor': 1,
                'K_Ea': -3, #log10(Ea)
                'ksr_Ea': -3,
                'KA_Ea': -3,
                'KB_Ea': -3}

temperature = 100 #K
rate_2 = general_rate(C_test, para_dict_2, temperature)

# Test on the ode solver
# Numerical integration step
ans_vec = ode_solver(dcdt, C0, t0, tf, stoichiometry, general_rate, para_dict_2, temperature)
Cf = ans_vec[-1, 1:] 

# Compute the final percentage conversion
# Use N2 concentrations
xf_2 = (C0[0] - Cf[0])/C0[0] * 100

# Compute the final rates
dcdt_f_2 = dcdt(tf, Cf, stoichiometry, general_rate, para_dict_2, temperature)

# Test on the reactor class
# xf_reactor should == xf
reactor_test_2 = Reactor(stoichiometry, P0, feed_composition, tf, temperature=temperature)
xf_reactor_2, _ = reactor_test_2.get_conversion(general_rate, para_dict_2)


