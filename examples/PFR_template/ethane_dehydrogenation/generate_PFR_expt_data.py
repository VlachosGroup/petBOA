"""
Generate PFR reactor data for ethane dehydrogenation
"""
import os
import sys
# temporary add the project path
# project_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
# sys.path.insert(0, project_path)
import numpy as np
import pmutt.constants as c
import pandas as pd
from estimator.reactor import Reactor

# %% Define the problem
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
n_specs = 7

# Energy unit used in the rate expression
energy_unit = 'kJ'
R_unit = 'kJ/mol/K'
R_bar_unit = 'L bar/mol/K'

# Inlet Concentration Vals in M
C2H6_in = 0.0005
C2H4_in = 0
CH4_in = 0
H2_in = 0
CO2_in = 0.0005
CO_in = 0
H2O_in = 0
C0 = [C2H6_in, C2H4_in, CH4_in, H2_in, CO2_in, CO_in, H2O_in]

# Set ground truth parameter values
A0_EDH = 2.5E6
Ea_EDH = 125  # kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110  # kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70  # kJ/mol

# Kp values calculated at different T 
# Import Kp data from an excel sheet 
Kp_file = 'Kp_Catalog.xlsx'
Kp_file_path = os.path.join(os.getcwd(), Kp_file)
Kp_data = pd.read_excel(Kp_file_path)
print(Kp_data)
# select the possible temperature values
T_column_name = 'Temperature'
temperatures = list(Kp_data['Temperature'])  # in K
nT = len(temperatures)

# default kinetic parameters to used to generate data
para_ethane = {'EDH_prefactor': np.log10(A0_EDH),
               'EDH_Ea': Ea_EDH,
               'Hyd_prefactor': np.log10(A0_Hyd),
               'Hyd_Ea': Ea_Hyd,
               'RWGS_prefactor': np.log10(A0_RWGS),
               'RWGS_Ea': Ea_RWGS
               }

# kinetic parameter names
para_name_ethane = list(para_ethane.keys())

# kinetic parameter values
para_ground_truth = list(para_ethane.values())


def rate_eq(concentrations, para_dict, rate_stoichiometry, name, temperature):
    """
    Rate equation involve pre-factor, Ea and
    KEQ (equilibrium constant) and Qp (reaction quotient)
    """
    # Convert to np array
    concentrations = np.array(concentrations, dtype=float)
    rate_stoichiometry = np.array(rate_stoichiometry)

    # Get the constants
    R_bar = c.R(R_bar_unit)
    R = c.R(R_unit)

    # Extract the parameters
    prefactor = 10 ** para_dict[name + '_prefactor']
    Ea = para_dict[name + '_Ea']
    KEQ = float(Kp_data[Kp_data['Temperature'] == temperature][name])

    # Reaction quotient 
    Qp = 0
    # Find the concentrations of the reactants
    reactant_index = np.where(rate_stoichiometry < 0)[0]
    participant_index = np.where(rate_stoichiometry != 0)[0]
    C_reactant_prod = np.prod(np.power(concentrations[reactant_index], - rate_stoichiometry[reactant_index]))

    if C_reactant_prod != 0:
        Qp = np.prod(np.power(concentrations[participant_index], rate_stoichiometry[participant_index])) \
             * np.power(temperature * R_bar, np.sum(rate_stoichiometry))

    rate_value = prefactor * np.exp(-Ea / R / temperature) * C_reactant_prod * (1 - Qp / KEQ)
    return rate_value


# PFR is integrated as a function of residence-time tau as opposed to length or volume
# therefore integration parameters are specified in residence time (in seconds)

# simulate the reactor at three residence times between 1 and 12 seconds
tau_list = list(np.linspace(1, 12, 3))

data = []
P = 1  # atm
# Generate the training data
for i, Ti in enumerate(temperatures):
    for tf_j in tau_list:
        reactor_ethane_Ti = Reactor(stoichiometry, tf_j, C0=C0, names=rxn_names, temperature=Ti)
        Cf = reactor_ethane_Ti.get_exit_concentration(rate_eq, para_ethane)
        # Parse the specifics (reaction conditions) of the reactor run
        # at inlet t = 0 seconds
        _list = [tf_j, Ti, P]
        for entry in C0:
            _list.append(entry)
        data.append(_list)
        # at the exit t = tau seconds
        _list = [tf_j, Ti, P]
        for entry in Cf: #exit concentration
            _list.append(entry)
        data.append(_list)

run_names = ["run_" + str(i) for i in range(len(temperatures)*len(tau_list))]
time_step_str = [0, 1] # time_step = t/tau
ind_names = pd.MultiIndex.from_product([run_names, time_step_str])
col_names = ["Residence time(s)", "Temperature(K)", "Pressure(atm)", "y_C2H6(M)", "y_C2H4(M)", "y_CH4(M)", "y_H2(M)",
             "y_CO2(M)", "y_CO(M)", "y_H2O(M)"]
Data = np.array(data).reshape(len(temperatures) * len(tau_list) * 2, len(col_names))
df = pd.DataFrame(Data, index=ind_names, columns=col_names)
df = df.rename_axis(index=("run_no", "time/tau"))
df.to_csv("ethane_pfr_data.csv")
