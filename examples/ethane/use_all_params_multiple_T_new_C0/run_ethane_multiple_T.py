"""
Test on ethane system ODEs
Test on the PFR case
"""
import os
import time
import sys
# temporary add the project path
project_path = os.path.abspath(os.path.join(os.getcwd(), '..\..\..'))
sys.path.insert(0, project_path)


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pmutt.constants as c
import pandas as pd

import estimator.utils as ut
from estimator.optimizer import ModelBridge, BOOptimizer
from estimator.reactor import Reactor

# Set matplotlib default values
font = {'size': 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2


def plot_ethane(t_vec, C_profile, title=None, save_path=None):
    """
    plot the ode profiles 
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_vec, C_profile[:, 0], label=r'$\rm C_{2}H_{6}$')
    ax.plot(t_vec, C_profile[:, 1], label=r'$\rm C_{2}H_{4}$')
    ax.plot(t_vec, C_profile[:, 2], label=r'$\rm CH_{4}$')
    ax.plot(t_vec, C_profile[:, 3], label=r'$\rm H_{2}$')
    ax.plot(t_vec, C_profile[:, 4], label=r'$\rm CO_{2}$')
    ax.plot(t_vec, C_profile[:, 5], label=r'$\rm CO$')
    ax.plot(t_vec, C_profile[:, 6], label=r'$\rm H_{2}O$')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('C (mol/L)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig_name = 'profile'
    if title is not None:
        ax.set_title(title)
        fig_name += ('_' + title.replace(' ', ''))
    if save_path is None:
        save_path = os.getcwd()

    fig.savefig(os.path.join(save_path, fig_name + '.png'), bbox_inches="tight")


def plot_ethane_overlap(t_vec1, C_profile1, t_vec2, C_profile2, title=None, save_path=None):
    """
    Plot the ode profiles, 
    Check whether the first profile matches with the second
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_vec1, C_profile1[:, 0], label=r'$\rm C_{2}H_{6}$')
    ax.plot(t_vec1, C_profile1[:, 1], label=r'$\rm C_{2}H_{4}$')
    ax.plot(t_vec1, C_profile1[:, 2], label=r'$\rm CH_{4}$')
    ax.plot(t_vec1, C_profile1[:, 3], label=r'$\rm H_{2}$')
    ax.plot(t_vec1, C_profile1[:, 4], label=r'$\rm CO_{2}$')
    ax.plot(t_vec1, C_profile1[:, 5], label=r'$\rm CO$')
    ax.plot(t_vec1, C_profile1[:, 6], label=r'$\rm H_{2}O$')
    # scatter for the second profile
    ax.scatter(t_vec2, C_profile2[:, 0], s=35, alpha=0.3)
    ax.scatter(t_vec2, C_profile2[:, 1], s=35, alpha=0.3)
    ax.scatter(t_vec2, C_profile2[:, 2], s=35, alpha=0.3)
    ax.scatter(t_vec2, C_profile2[:, 3], s=35, alpha=0.3)
    ax.scatter(t_vec2, C_profile2[:, 4], s=35, alpha=0.3)
    ax.scatter(t_vec2, C_profile2[:, 5], s=35, alpha=0.3)
    ax.scatter(t_vec2, C_profile2[:, 6], s=35, alpha=0.3)

    ax.set_xlabel('t (s)')
    ax.set_ylabel('C (mol/L)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig_name = 'profiles_overlap'
    if title is not None:
        ax.set_title(title)
        fig_name += ('_' + title.replace(' ', ''))
    if save_path is None:
        save_path = os.getcwd()

    fig.savefig(os.path.join(save_path, fig_name + '.png'), bbox_inches="tight")


def plot_ethane_residual(t_vec1, C_profile1, t_vec2, C_profile2, title=None, save_path=None):
    """
    Plot the ode profiles,
    Check whether the first profile matches with the second
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_vec1, C_profile1[:, 0] - C_profile2[:, 0], label=r'$\rm C_{2}H_{6}$')
    ax.plot(t_vec1, C_profile1[:, 1] - C_profile2[:, 1], label=r'$\rm C_{2}H_{4}$')
    ax.plot(t_vec1, C_profile1[:, 2] - C_profile2[:, 2], label=r'$\rm CH_{4}$')
    ax.plot(t_vec1, C_profile1[:, 3] - C_profile2[:, 3], label=r'$\rm H_{2}$')
    ax.plot(t_vec1, C_profile1[:, 4] - C_profile2[:, 4], label=r'$\rm CO_{2}$')
    ax.plot(t_vec1, C_profile1[:, 5] - C_profile2[:, 5], label=r'$\rm CO$')
    ax.plot(t_vec1, C_profile1[:, 6] - C_profile2[:, 6], label=r'$\rm H_{2}O$')

    ax.set_xlabel('t (s)')
    ax.set_ylabel('C (mol/L)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig_name = 'error_residual'
    if title is not None:
        ax.set_title(title)
        fig_name += ('_' + title.replace(' ', ''))
    if save_path is None:
        save_path = os.getcwd()

    fig.savefig(os.path.join(save_path, fig_name + '.png'), bbox_inches="tight")


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
m_specs = 7

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

# varying integration times
n_tf = 3
tmax = 12
tf = list(np.linspace(1, tmax, n_tf))
t_eval = np.linspace(0, tmax, 100)

# Set ground truth parameter values
A0_EDH = 2.5E6
Ea_EDH = 125  # kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110  # kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70  # kJ/mol

# Kp values calculated at different T 
# Import Kp data
Kp_file = 'Kp_Catalog.xlsx'
example_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
Kp_file_path = os.path.join(example_path, Kp_file)
Kp_data = pd.read_excel(Kp_file_path)
# select the possible temperature values
T_column_name = 'Temperature'
temperatures = list(Kp_data['Temperature']) # in K
nT = len(temperatures)

# 6 parameters to fit
para_ethane = {'EDH_prefactor': np.log10(A0_EDH),
               'EDH_Ea': Ea_EDH,
               'Hyd_prefactor': np.log10(A0_Hyd),
               'Hyd_Ea': Ea_Hyd,
               'RWGS_prefactor': np.log10(A0_RWGS),
               'RWGS_Ea': Ea_RWGS
               }

# parameter names
para_name_ethane = list(para_ethane.keys())

# ground truth parameter values
para_ground_truth = list(para_ethane.values())

# Estimator name
estimator_name = 'ethane_multiple_T_PFR'
ut.clear_cache(estimator_name)


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
    prefactor = 10 ** para_dict[name + '_prefactor']
    Ea = para_dict[name + '_Ea']
    KEQ = float(Kp_data[Kp_data['Temperature']==temperature]['EDH'])

    # Reaction quotient 
    Qp = 0
    # Find the concentrations of the reactants
    reactant_index = np.where(stoichiometry < 0)[0]
    participant_index = np.where(stoichiometry != 0)[0]
    C_reactant_prod = np.prod(np.power(concentrations[reactant_index], - stoichiometry[reactant_index]))

    if C_reactant_prod != 0:
        Qp = np.prod(np.power(concentrations[participant_index], stoichiometry[participant_index])) \
             * np.power(temperature * R_bar, np.sum(stoichiometry))

    rate_value = prefactor * np.exp(-Ea/R/temperature) * C_reactant_prod * (1 - Qp / KEQ)
    return rate_value


# %% Tests on the ethane system at different temperatures
# Generate the ground truth training data
reactor_data = []
Y_experiments = []
Y_experiments_constant_T = [[] for _ in range(nT)]# a vector for plotting purposes
Y_weights = np.ones(m_specs)
# Set the weight for water as zero, others as 1
Y_weights[-1] = 0.

for i, Ti in enumerate(temperatures):        
    for tf_j in tf:
        reactor_ethane_Ti = Reactor(stoichiometry, tf_j, C0=C0, names=rxn_names, temperature=Ti)
        Cf = reactor_ethane_Ti.get_exit_concentration(rate_eq, para_ethane)
        
        # Test the model bridge
        # Parse the specifics (reaction conditions) of reactor object into a dictionary
        reactor_i = {}
        reactor_i['C0'] = C0
        reactor_i['tf'] = tf_j
        reactor_i['names'] = rxn_names
        reactor_i['temperature'] = Ti
        reactor_data.append(reactor_i)
        # Use conversion as the "experimental" data, n_reactor = nT * n_tf
        Y_experiment_ij = Cf
        Y_experiments.append(Y_experiment_ij)
        Y_experiments_constant_T[i].append(Y_experiment_ij)
    
    

#%% Input experimental data and models (rate expressions) into a model bridge
bridge = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
bridge.input_data(stoichiometry, reactor_data, Y_experiments, Y_weights,  qoi='concentration')


#%% Set up an optimizer
# Automatically compute the parameter ranges given a deviation
deviation = 0.25
para_ranges = [[-np.abs(vi) * deviation + vi, np.abs(vi) * deviation + vi] for vi in para_ground_truth]

# Start a timer
start_time = time.time()
n_iter = 100
optimizer = BOOptimizer(estimator_name)
X_opt, loss_opt, Exp = optimizer.optimize(bridge.loss_func, para_ranges, n_iter, log_flag=True)
end_time = time.time()



#%% 
para_ethane_opt = {}
for xi, name_i in zip(X_opt, para_name_ethane):
    para_ethane_opt[name_i] = xi
t_opt = []
Y_opt = []

# Predict the conversions given the optimal set of parameters
for i, Ti in enumerate(temperatures):
    reactor_ethane_Ti = Reactor(stoichiometry, tmax, C0=C0, names=rxn_names, temperature=Ti)
    tC_profile = reactor_ethane_Ti.get_profile(rate_eq, para_ethane_opt, t_eval = t_eval)
    t_opt_i = tC_profile[:, 0]
    Y_opt_i = tC_profile[:, 1:]
    t_opt.append(t_opt_i)
    Y_opt.append(Y_opt_i)
    plot_ethane_overlap(t_opt_i, Y_opt_i, tf, np.array(Y_experiments_constant_T[i]), 'Optimal Set @ T=' + str(Ti), estimator_name)


#%%
# Print the results
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt, para_ground_truth)
