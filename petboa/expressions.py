"""
Rate expressions only for A+B=R (ABtoR)
"""
import numpy as np
from pmutt import constants as c

species_names = ['A', 'B', 'R']

#%% Define the form of the rate constant
class RateConstant():
    def __init__(self, name = 'k'):
        self.name = name
        
    def value(self, para_dict, temperature=None, energy_unit='eV'):
        if temperature is None:
            k_value = para_dict[self.name]  # input is log10(prefactor)
        else:
            # based on the unit of Ea, must be J, kJ, cal, kcal, eV etc.
            # set the unit for kb
            kb_unit = energy_unit + '/K'
                
            prefactor = para_dict[self.name+'_prefactor']
            Ea = 10**(para_dict[self.name+'_Ea']) # input is log10(Ea)
            k_value = prefactor * np.exp(-Ea/c.kb(kb_unit)/temperature)
            
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
def general_rate(concentrations, para_dict, stoichiometry=None, name=None, temperature=None):
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

def general_rate_ads(concentrations, para_dict, stoichiometry=None, name=None, temperature=None):
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


