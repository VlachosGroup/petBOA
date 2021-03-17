# -*- coding: utf-8 -*-
"""
Rate expressions only for A+B=R (ABtoR)
"""
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint, ode

# concentrations = {'A': 1,
#                 'B': 1,
#                 'R': 0}
all_species = ['A', 'B', 'R']

parameters_1 = {'K': 1,
                'ksr': 1,
                'KA': 1,
                'KB': 1}

parameters_2 = {'k1': 1,
               'k2': 1,
               'alpha': 1,
               'beta': 1}


# Driving force group (DFG)
def driving_suface_reaction_controlling(concentrations, parameters):
    return concentrations[0]*concentrations[1] - concentrations[2]/parameters['K']

driving_force_groups = {'surface reaction controlling': driving_suface_reaction_controlling}

# Kinetic group
def kinetic_suface_reaction_controlling(parameters):
    return parameters['ksr']*parameters['KA']*parameters['KB']

kinetic_groups = {'surface reaction controlling': kinetic_suface_reaction_controlling}

# Adsorption group
def adsorption_equilirium_w_dissociation(concentrations, parameters, species = 'A'):
    return np.sqrt(parameters['K'+ species]*concentrations[all_species.index(species)])

adsorption_groups = {'equilirium adsorption with dissociation': adsorption_equilirium_w_dissociation}


# Exponents of adsorption groups
exponents = {'surface reaction controlling': {'dissociation': 3}}


# General rate experssion
def general_rate(concentrations, parameters):
    # Putting pieces together
    controling_key = 'surface reaction controlling'
    ads_key = 'equilirium adsorption with dissociation'
    surface_reaction_key = 'dissociation'
    
    adsorption_terms = (1 + adsorption_groups[ads_key](concentrations, parameters, 'A') + \
        adsorption_groups[ads_key](concentrations, parameters, 'B'))**exponents[controling_key][surface_reaction_key]
    
    rate = driving_force_groups[controling_key](concentrations, parameters) * \
        kinetic_groups[controling_key](parameters)/adsorption_terms
    
    return rate


# Temkin Pyzhev expression
def temkin_pyzhev_rate(concentrations, parameters):
    rate = parameters['k1']*concentrations[0]*(concentrations[1]**3/concentrations[0]**2)**parameters['alpha'] -\
        parameters['k2']*(concentrations[2]**2/concentrations[1]**3)**parameters['beta']

    return rate
    

def species_dcdt(t, concentrations, rate_expression, parameters):
    
    cur_rate = rate_expression(concentrations, parameters)

    # dC/dt
    dcdt = np.zeros(3)
    dcdt[0] = -cur_rate
    dcdt[1] = -3*cur_rate
    dcdt[2] = 2*cur_rate
    
    return dcdt
    
    #stoichiometry can be an input

def ode_solver(func, concentrations0, t0, tf, *args):
    # set up ode solver
    # Construct the ode solver, ode45 with varying step size
    sol = []
    def solout(t, y):
        sol.append([t, *y])
        
    solver = ode(func).set_integrator('dopri5', rtol  = 1e-6, method='bdf')
    solver.set_solout(solout)
    #feed in argumenrs and initial conditions for odes
    solver.set_initial_value(concentrations0, t0).set_f_params(*args) 
    solver.integrate(tf)
    sol = np.array(sol)
    
    return sol
    
    
#%% Test on rates
# rate should be 1/27
C_test = np.array([100,300,0])
rate_1 = general_rate(C_test, parameters_1)
rate_2 = temkin_pyzhev_rate(C_test, parameters_2)

#%% 
C0 = np.array([100,300,0])
t0 = 0 
tf = 100

sol = ode_solver(species_dcdt, C0, t0, tf, general_rate, parameters_1)
