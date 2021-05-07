"""
Test on ethane system ODEs
"""
import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_path)

import time
import numpy as np
import pmutt.constants as c

from estimator.reactor import Reactor
from estimator.optimizer import ModelBridge, BOOptimizer

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

#%% Define the problem 
# Reaction equations:
# EDH: ethane dehydrogenation: C2H6 -> C2H4 + H2
# Hyd: hydrogenolysis: C2H6 + H2 -> 2CH4
# RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O

# stoichiometry matrix
# C2H6, C2H4, CH4, H2, CO2, CO, H2O
stoichiometry = [[-1, 1, 0, 1, 0, 0, 0], 
                 [-1, 0, 2, -1, 0, 0, 0],
                 [0, 0, 0, -1, -1, 1, 1]]

# Set elementary reaction names
rxn_names = ['EDH', 'Hyd', 'RWGS']

# number of species 
m_specs = 7

# Energy unit used in the rate expression
energy_unit = 'kJ'
R_unit = 'kJ/mol/K'
R_bar_unit = 'L bar/mol/K'
temperature = 873 #K

# Inlet Concentration Vals in M
C2H6_in = 0.0005
C2H4_in = 0
CH4_in = 0
H2_in = 0
CO2_in = 0.0005
CO_in = 0
H2O_in = 0
C0 = [C2H6_in, C2H4_in, CH4_in, H2_in, CO2_in, CO_in, H2O_in]


# Integration time
tf = 10

# we can set the integration t_eval (optional)
n_int = 101
t_eval = np.linspace(0, tf, n_int)

# Set ground truth parameter values
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

# Ground truth parameter
para_ethane = {'EDH_prefactor': np.log10(A0_EDH),
                'EDH_Ea': Ea_EDH,
                'EDH_eq': np.log10(Kp_EDH[temperature]),
                'Hyd_prefactor': np.log10(A0_Hyd),
                'Hyd_Ea': Ea_Hyd,
                'Hyd_eq': np.log10(Kp_Hyd[temperature]),
                'RWGS_prefactor': np.log10(A0_RWGS),
                'RWGS_Ea': Ea_RWGS,
                'RWGS_eq': np.log10(Kp_RWGS[temperature])}

# parameter names
para_name_ethane = list(para_ethane.keys())

# ground truth parameter values
para_ground_truth = list(para_ethane.values())


def rate_eq(concentrations, para_dict, stoichiometry, name, temperature):
    """
    Rate equation involve prefactor, Ea and 
    KEQ (equilirium constant) and Qp (reaction quotient)
    """
    # Convert to nparray
    concentrations = np.array(concentrations, dtype=float)
    stoichiometry = np.array(stoichiometry)
    
    # Get the constants
    R_bar = c.R(R_bar_unit)
    R = c.R(R_unit)
    
    # Extract the parameters
    prefactor = 10**para_dict[name+'_prefactor']
    Ea = para_dict[name+'_Ea']
    KEQ = 10**para_dict[name+'_eq']
    
    # Reaction quotient 
    Qp = 0
    # Find the concentrations of the reactants
    reactant_index = np.where(stoichiometry < 0)[0]
    participant_index = np.where(stoichiometry != 0)[0]
    C_reactant_prod = np.prod(np.power(concentrations[reactant_index], - stoichiometry[reactant_index])) 
    
    if C_reactant_prod != 0:
        Qp = np.prod(np.power(concentrations[participant_index], stoichiometry[participant_index]))\
            * np.power(temperature * R_bar, np.sum(stoichiometry)) 
    
    rate_value = prefactor * np.exp(-Ea/R/temperature) * C_reactant_prod * (1 - Qp/KEQ)
        
    return rate_value


#%%
# Test the model bridge
# Prase the specifics (reaction conditions) of reactor object into a dictionary
reactor_i = {}
reactor_i['C0'] = C0
reactor_i['tf'] = tf
reactor_i['names'] = rxn_names
reactor_i['temperature'] = temperature
reactor_data = [reactor_i]



#%%
from estimator.optimizer import ParameterMask

para_range_x = [[0,1], [0, 10], 5, 6, [7, 8]]
mask = ParameterMask(para_range_x)

X_full = mask.prepare_X([[0, 0, 7], [1, 10, 8]])
