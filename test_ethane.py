"""
Test on ethane system ODEs
"""

import numpy as np
import pmutt.constants as c
from reactor import Reactor


energy_unit = 'kJ'
R_unit = 'kJ/mol/K'
R_bar_unit = 'L bar/mol/K'
temperature = 873 #K

#Set parameter values
A0_EDH = 2.5E6
Ea_EDH = 125 #kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110 #kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70 #kJ/mol

# Kp values calculated at different T 
# T kp value pair in a dictionary
Kp_EDH = {835: 0.0114,
          848: 0.0157,
          861: 0.0214,
          873: 0.0281}
Kp_Hyd = {835: 28700,
          848: 24500,
          861: 21100,
          873: 18400}
Kp_RWGS = {835: 0.296,
           848: 0.321,
           861: 0.347,
           873: 0.372}


para = {'EDH_prefactor': A0_EDH,
        'EDH_Ea': Ea_EDH,
        'EDH_eq': Kp_EDH[temperature],
        'Hyd_prefactor': A0_Hyd,
        'Hyd_Ea': Ea_Hyd,
        'Hyd_eq': Kp_Hyd[temperature],
        'RWGS_prefactor': A0_RWGS,
        'RWGS_Ea': Ea_RWGS,
        'RWGS_eq': Kp_RWGS[temperature]}


# Inlet Concentration Vals in M
C2H6_in = 0.0005
C2H4_in = 0
CH4_in = 0
H2_in = 0
CO2_in = 0.0005
CO_in = 0
H2O_in = 0
C0 = [C2H6_in, C2H4_in, CH4_in, H2_in, CO2_in, CO_in, H2O_in]


# Reaction equations:
# EDH: ethane dehydrogenation: C2H6 -> C2H4 + H2
# Hyd: hydrogenolysis: C2H6 + H2 -> 2CH4
# RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O

# stoichiometry matrix
# C2H6, C2H4, CH4, H2, CO2, CO, H2O
stoichiometry = [[-1, 1, 0, 1, 0, 0, 0], 
                 [-1, 0, 2, -1, 0, 0, 0],
                 [0, 0, 0, -1, -1, 1, 1]]


# Convert to nparray
y_test = [1, 2, 3, 4, 5, 6, 7]


#%% Define a new form of the rate constant with equilirium constant
def rate_eq(concentrations, stoichiometry, para_dict = None, name = 'r',  temperature=300):
    
    # Convert to nparray
    concentrations = np.array(concentrations, dtype=float)
    stoichiometry = np.array(stoichiometry)
    
    # Get the constants
    R_bar = c.R(R_bar_unit)
    R = c.R(R_unit)
    
    #Extract the parameters
    prefactor = para_dict[name+'_prefactor']
    Ea = para_dict[name+'_Ea']
    KEQ = para_dict[name+'_eq']
    
    Qp = 0
    # find the concentrations of the reactants
    reactant_index = np.where(stoichiometry < 0)[0]
    participant_index = np.where(stoichiometry != 0)[0]
    
    C_reactant_prod = np.prod(np.power(concentrations[reactant_index], - stoichiometry[reactant_index])) 
    
    if C_reactant_prod != 0:
        Qp = np.prod(np.power(concentrations[participant_index], stoichiometry[participant_index]))\
            * np.power(temperature * R_bar, np.sum(stoichiometry)) 
    
    rate_value = prefactor * np.exp(-Ea/R/temperature) * C_reactant_prod * (1 - Qp/KEQ)
        
    return rate_value
    
rate_EDH = rate_eq(y_test, stoichiometry[0], para_dict=para, name='EDH', temperature=temperature)
rate_Hyd = rate_eq(y_test, stoichiometry[1], para_dict=para, name='Hyd', temperature=temperature)
rate_RWGS = rate_eq(y_test, stoichiometry[2], para_dict=para, name='RWGS', temperature=temperature)

reactor_EDH = Reactor(stoichiometry, P0, feed_composition, tf, temperature=temperature)




