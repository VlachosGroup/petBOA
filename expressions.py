"""
Rate expressions only for A+B=R (ABtoR)
"""
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint, ode

#%% Inputs for the current reaction
species_names = ['A', 'B', 'R']
stoichiometry = [-1, -3, 2]

# Dictionaries for parameters
para_dict_1 = {'K': 1,
                'ksr': 1,
                'KA': 1,
                'KB': 1}

para_dict_2 = {'k1': 1,
                'k2': 1,
                'alpha': 1,
                'beta': 1}

#%% Define all groups in the table as dictionaries
# Driving force group (DFG)
def driving_suface_reaction_controlling(concentrations, para_dict):
    return concentrations[0]*concentrations[1] - concentrations[2]/para_dict['K']

driving_force_groups = {'surface reaction controlling': driving_suface_reaction_controlling}

# Kinetic group
def kinetic_suface_reaction_controlling(para_dict):
    return para_dict['ksr']*para_dict['KA']*para_dict['KB']

kinetic_groups = {'surface reaction controlling': kinetic_suface_reaction_controlling}

# Adsorption group
def adsorption_equilirium_w_dissociation(concentrations, para_dict, species = 'A'):
    return np.sqrt(para_dict['K'+ species]*concentrations[species_names.index(species)])

adsorption_groups = {'equilirium adsorption with dissociation': adsorption_equilirium_w_dissociation}

# Exponents of adsorption groups
exponents = {'surface reaction controlling': {'dissociation': 3}}


#%% Define the rate expressions
# General rate experssion
def general_rate(concentrations, para_dict):
    """Rate expressions from Yang and Hougen
    """
    controling_key = 'surface reaction controlling'
    ads_key = 'equilirium adsorption with dissociation'
    surface_reaction_key = 'dissociation'
    
    adsorption_terms = (1 + adsorption_groups[ads_key](concentrations, para_dict, 'A') + \
        adsorption_groups[ads_key](concentrations, para_dict, 'B'))**exponents[controling_key][surface_reaction_key]
    
    rate = driving_force_groups[controling_key](concentrations, para_dict) * \
        kinetic_groups[controling_key](para_dict)/adsorption_terms
    
    return rate


def temkin_pyzhev_rate(concentrations, para_dict):
    """Rate expression for ammonia chemsitry from Temkin and Pyzhev
    """
    rate = para_dict['k1']*concentrations[0]*(concentrations[1]**3/concentrations[0]**2)**para_dict['alpha'] -\
        para_dict['k2']*(concentrations[2]**2/concentrations[1]**3)**para_dict['beta']

    return rate
    

#%% ODE functions for numerical integration
def dcdt(t, concentrations, rate_expression, stoichiometry, para_dict):
    """Compute the derivatives
    """
    cur_rate = rate_expression(concentrations, para_dict)
    n_spec = len(stoichiometry)

    # dC/dt for each species
    cur_dcdt = np.zeros(n_spec)
    for i in range(n_spec):
        cur_dcdt[i] = stoichiometry[i] * cur_rate
 
    return cur_dcdt
    

def ode_solver(func, y0, t0, tf, *args):
    """Set up the ode solver 
    """
    # Construct the ode solver, ode45 with varying step size
    ans = []
    def get_ans(t, y):
        ans.append([t, *y])
        
    solver = ode(func).set_integrator('dopri5', rtol  = 1e-6, method='bdf')
    solver.set_solout(get_ans)
    #feed in argumenrs and initial conditions for odes
    solver.set_initial_value(y0, t0).set_f_params(*args) 
    solver.integrate(tf)
    ans = np.array(ans)
    
    return ans
    
    
#%% Test on rates given a set of concentrations
# rate should be 1/27
C_test = np.array([100,300,0])
rate_1 = general_rate(C_test, para_dict_1)
rate_2 = temkin_pyzhev_rate(C_test, para_dict_2)

#%% Test on the ode solver
C0 = np.array([100,300,0])
t0 = 0 
tf = 100

# Numerical integration step
ans = ode_solver(dcdt, C0, t0, tf, general_rate, stoichiometry, para_dict_1)
