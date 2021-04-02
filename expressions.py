"""
Rate expressions only for A+B=R (ABtoR)
"""
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint, ode

species_names = ['A', 'B', 'R']

from pmutt import constants as c

class RateConstant():
    def __init__(self, name = 'k'):
        self.name = name
        
    def value(self, para_dict, temperature=None):
        if temperature is None:
            k_value = para_dict[self.name]  # input is log10(prefactor)
        else:
            prefactor = para_dict[self.name+'_prefactor']
            Ea = 10**(para_dict[self.name+'_Ea']) # input is log10(Ea)
            k_value = prefactor * np.exp(-Ea/c.kb('eV/K')/temperature)
            
        return k_value
    
        
#%% Define all groups in the table as dictionaries
#%%
# Driving force group (DFG)
def driving_suface_reaction_controlling(concentrations, para_dict, temperature=None):
    
    K = RateConstant('K').value(para_dict, temperature)
    return concentrations[0]*concentrations[1] - concentrations[2]/K

def driving_adsorption_controlling_w_dissociation(concentrations, para_dict, temperature=None):
    
    K = RateConstant('K').value(para_dict, temperature)
    return concentrations[0] - concentrations[2]/concentrations[1]/K

driving_force_groups = {'surface reaction controlling': driving_suface_reaction_controlling, 
                        'adsorption controlling': driving_adsorption_controlling_w_dissociation}


#%%
# Kinetic group
def kinetic_suface_reaction_controlling(para_dict, temperature=None):

    ksr = RateConstant('ksr').value(para_dict, temperature)
    KA = RateConstant('KA').value(para_dict, temperature)
    KB = RateConstant('KB').value(para_dict, temperature)
    
    return ksr*KA*KB

def kinetic_adsorption_controlling_w_dissociation(para_dict, species = 'A', temperature=None):

    KA = RateConstant('K'+species).value(para_dict, temperature)
    
    return KA

kinetic_groups = {'surface reaction controlling': kinetic_suface_reaction_controlling, 
                  'adsorption controlling with dissociation': kinetic_adsorption_controlling_w_dissociation}

#%%
# Adsorption group

def adsorption_default(concentrations, para_dict, species = 'A', temperature=None):
    Kx = RateConstant('K'+species).value(para_dict, temperature)
    return Kx*concentrations[species_names.index(species)]

def adsorption_equilirium_w_dissociation(concentrations, para_dict, species = 'A', temperature=None):
    
    Kx = RateConstant('K'+species).value(para_dict, temperature)
    return np.sqrt(Kx*concentrations[species_names.index(species)])

def adsorption_controlling_w_dissociation(concentrations, para_dict, species = 'A', temperature=None):
    
    Kx = RateConstant('K'+species).value(para_dict, temperature)
    K = RateConstant('K').value(para_dict, temperature)
    return np.sqrt(Kx*concentrations[species_names.index('R')]/K/concentrations[species_names.index('B')])

adsorption_groups = {'adsorption default': adsorption_default,
                     'adsorption equilirium with dissociation': adsorption_equilirium_w_dissociation,
                     'adsorption controlling with dissociation': adsorption_controlling_w_dissociation}

# Exponents of adsorption groups
exponents = {'surface reaction controlling': {'dissociation': 3},
             'adsorption controlling with dissociation': 2}

#%% Define the rate expressions
# General rate experssion
def general_rate(concentrations, para_dict, temperature=None):
    """Rate expressions from Yang and Hougen
    """
    controling_key = 'surface reaction controlling'
    ads_key = 'adsorption equilirium with dissociation'
    surface_reaction_key = 'dissociation'
    
    adsorption_terms = (1 + adsorption_groups[ads_key](concentrations, para_dict, 'A', temperature) + \
        adsorption_groups[ads_key](concentrations, para_dict, 'B', temperature))**exponents[controling_key][surface_reaction_key]
    
    rate = driving_force_groups[controling_key](concentrations, para_dict, temperature) * \
        kinetic_groups[controling_key](para_dict, temperature)/adsorption_terms
    
    return rate

def general_rate_ads(concentrations, para_dict, temperature=None):
    """Rate expressions from Yang and Hougen
    """
    controling_key = 'adsorption controlling'
    ads_key = 'adsorption controlling with dissociation'
    #surface_reaction_key = 'dissociation'
    
    adsorption_terms = (1 + adsorption_groups[ads_key](concentrations, para_dict, 'A', temperature) + \
        adsorption_groups['adsorption default'](concentrations, para_dict, 'B', temperature))**exponents[ads_key]
    
    rate = driving_force_groups[controling_key](concentrations, para_dict, temperature) * \
        kinetic_groups[ads_key](para_dict, temperature)/adsorption_terms
    
    return rate

#%% ODE functions for numerical integration
def dcdt(t, concentrations, stoichiometry, rate_expression,  para_dict, temperature=None):
    """Compute the derivatives
    """
    cur_rate = rate_expression(concentrations, para_dict, temperature)
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
    

class Reactor():
    """Reaction ODEs class"""
    
    def __init__(self, stoichiometry, P0, feed_composition, tf, name = 'simple reaction', temperature=None):
        """Initialize the constants"""
        self.stoichiometry = stoichiometry
        self.name = name
        self.P0 = P0
        self.feed_composition = feed_composition
        self.t0 = 0
        self.tf = tf
        self.temperature = temperature
        
        self.C0 = P0 * np.array(feed_composition)/np.sum(feed_composition)
    
    def get_profile(self, rate_expression, para_dict):
        """Numerical integration of the rate expression given the parameters"""
        
        tC_profile = ode_solver(dcdt, self.C0, self.t0, self.tf, self.stoichiometry, rate_expression, para_dict, self.temperature)
        
        return tC_profile
        
    def get_conversion(self, rate_expression, para_dict, species_index = 0):
        """Get the final conversion of a specie"""
        
        # Get the profile
        tC_profile = self.get_profile(rate_expression, para_dict)
        # Extract the final concentrations
        Cf = tC_profile[-1, species_index+1:] 
        # Compute the final percentage conversion
        xf = (self.C0[species_index] - Cf[species_index])/self.C0[species_index] * 100
        
        # Compute the final rates
        dcdt_f = dcdt(self.tf, Cf, self.stoichiometry, rate_expression, para_dict, self.temperature)
        
        return xf, dcdt_f
                 

