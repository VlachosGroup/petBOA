"""
Tests on expressions module
"""

import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import numpy as np
from estimator.reactor import Reactor, ode_solver, ode_solver_ivp, dcdt
from estimator.expressions import general_rate

import matplotlib.pyplot as plt
import matplotlib
# Set matplotlib default values
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2


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
ans_vec_ivp = ode_solver_ivp(dcdt, C0, t0, tf, None, 'LSODA', stoichiometry, general_rate, para_dict_1)

Cf = ans_vec_ivp[-1, 1:] 
#Cf = ans_vec

# Compute the final percentage conversion
# Use N2 concentrations
xf_1 = (C0[0] - Cf[0])/C0[0] * 100

# Compute the final rates
dcdt_f_1 = dcdt(tf, Cf, stoichiometry, general_rate, para_dict_1)
#dcdt_f_1_m = dcdt_m(tf, Cf, stoichiometry, general_rate, para_dict_1)

# Test on the reactor class
# xf_reactor should == xf
reactor_test_1 = Reactor(stoichiometry, tf, P0, feed_composition)
xf_reactor_1, _ = reactor_test_1.get_conversion(general_rate, para_dict_1)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(ans_vec[:,0], ans_vec[:,1], label = r'$\rm N_{2}$')
ax.plot(ans_vec[:,0], ans_vec[:,2], label = r'$\rm H_{2}$')
ax.plot(ans_vec[:,0], ans_vec[:,3], label = r'$\rm NH_{3}$')
ax.set_xlabel('t (s)')
ax.set_ylabel(r'$\rm P_{i}\ (atm)$')
ax.legend()

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
rate_2 = general_rate(C_test, para_dict_2, stoichiometry=stoichiometry, temperature=temperature)

# Test on the ode solver
# Numerical integration step
ans_vec = ode_solver_ivp(dcdt, C0, t0, tf, None, 'LSODA',stoichiometry, general_rate, para_dict_2, None, temperature)
Cf = ans_vec[-1, 1:] 

# Compute the final percentage conversion
# Use N2 concentrations
xf_2 = (C0[0] - Cf[0])/C0[0] * 100

# Compute the final rates
dcdt_f_2 = dcdt(tf, Cf, stoichiometry, general_rate, para_dict_2, None, temperature)

# Test on the reactor class
# xf_reactor should == xf
reactor_test_2 = Reactor(stoichiometry, tf, P0, feed_composition, temperature=temperature)
xf_reactor_2, _ = reactor_test_2.get_conversion(general_rate, para_dict_2)


