"""
Ethane dehydrogenation batch reactor model data
and the rate expression: Reaction equations:
EDH: ethane dehydrogenation: C2H6 -> C2H4 + H2
Hyd: hydrogenolysis: C2H6 + H2 -> 2CH4
RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O
"""
import numpy as np
import pmutt.constants as c

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

Kp = {"EDH": {835: 0.0114,
              848: 0.0157,
              861: 0.0214,
              873: 0.0281},
      "Hyd": {835: 28700,
              848: 24500,
              861: 21100,
              873: 18400},
      "RWGS": {835: 0.296,
               848: 0.321,
               861: 0.347,
               873: 0.372}
      }


def rate_eq(concentrations, para_dict, _stoichiometry, name, temperature):
    """
    Rate equation involve pre-factor, Ea and
    KEQ (equilibrium constant) and Qp (reaction quotient)
    """
    # Convert to np array
    concentrations = np.array(concentrations, dtype=float)
    rxn_stoichiometry = np.array(_stoichiometry)

    # Get the constants
    R_bar = c.R(R_bar_unit)
    R = c.R(R_unit)

    # Extract the parameters
    prefactor = 10 ** para_dict[name + '_prefactor']
    Ea = para_dict[name + '_Ea']
    KEQ = float(Kp[name][int(temperature)])

    # Reaction quotient
    Qp = 0.0
    # Find the concentrations of the reactants
    reactant_index = np.where(rxn_stoichiometry < 0)[0]
    participant_index = np.where(rxn_stoichiometry != 0)[0]
    c_reactant_prod = np.prod(np.power(concentrations[reactant_index], - rxn_stoichiometry[reactant_index]))

    if c_reactant_prod != 0:
        Qp = np.prod(np.power(concentrations[participant_index], rxn_stoichiometry[participant_index])) \
             * np.power(temperature * R_bar, np.sum(rxn_stoichiometry))

    rate_value = prefactor * np.exp(-Ea / R / temperature) * c_reactant_prod * (1 - Qp / KEQ)
    return rate_value
