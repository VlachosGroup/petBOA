"""
Test on ethane system ODEs
"""

import numpy as np
import pmutt.constants as c

energy_unit = 'kJ'
R_unit = 'kJ/mol/K'
R_bar_unit = 'L bar/mol/K'
temperature = 873 #K
#%% Define a new form of the rate constant with equilirium constant


class RateConstantEq():
    def __init__(self, name = 'r'):
        self.name = name
        
    def value(self, concentrations, stoichiometry, para_dict = None, temperature=1):
        
        # Convert to nparray
        concentrations = np.array(concentrations, dtype=float)
        stoichiometry = np.array(stoichiometry)
        
        # Get the constants
        R_bar = c.R(R_bar_unit)
        R = c.R(R_unit)
        
        # Extract the parameters
        # prefactor = para_dict[self.name+'_prefactor']
        # Ea = 10**(para_dict[self.name+'_Ea']) # input is log10(Ea)
        # KEQ = para_dict[self.name+'_eq']
        
        Qp = 0
        # find the concentrations of the reactants
        reactant_index = np.where(stoichiometry < 0)[0]
        participant_index = np.where(stoichiometry != 0)[0]
        
        Cprod_reactant = np.prod(np.power(concentrations[reactant_index], stoichiometry[reactant_index])) 
        
        if Cprod_reactant != 0:
            print(participant_index)
            Qp = np.prod(np.power(concentrations[participant_index], stoichiometry[participant_index]))\
                * np.power(temperature * R_bar, np.sum(stoichiometry)) 
        
        k_value = prefactor * np.exp(-Ea/R/temperature) * (1 - Qp)
            
        return Qp
    
    

