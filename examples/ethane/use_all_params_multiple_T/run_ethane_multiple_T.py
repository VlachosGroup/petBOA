"""
Test on ethane system ODEs
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
temperature = 873  # K

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
n_int = 11
t_eval = np.linspace(0, tf, n_int)

# Set ground truth parameter values
A0_EDH = 2.5E6
Ea_EDH = 125  # kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110  # kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70  # kJ/mol

# Kp values calculated at different T 
# T kp value pair in a dictionary

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
estimator_name = 'ethane_multiple_T'
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
    KEQ = Kp[name][int(temperature)]

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
# Test on whether the rate can be calculated correctly
reactor_data = []
Y_experiments = []
n_reactor = len(Kp["EDH"].keys())
Y_weights = np.ones(m_specs)
Y_weights[-1] = 0.
for T in Kp["EDH"].keys():
    temperature = float(T)
    rate_EDH = rate_eq(C0, para_ethane, stoichiometry[0], 'EDH', temperature)
    rate_Hyd = rate_eq(C0, para_ethane, stoichiometry[1], 'Hyd', temperature)
    rate_RWGS = rate_eq(C0, para_ethane, stoichiometry[2], 'RWGS', temperature)
    # Test on a reactor model to perform integration
    reactor_ethane = Reactor(stoichiometry, tf, C0=C0, names=rxn_names, temperature=temperature)
    tC_profile = reactor_ethane.get_profile(rate_eq, para_ethane, t_eval=t_eval)
    # Plot the profile
    t_eval = tC_profile[:, 0]
    C_profile = tC_profile[:, 1:]
    plot_ethane(t_eval, C_profile, 'Ground truth at ' + str(T) + ' (K)', estimator_name)
    # Test the model bridge
    # Parse the specifics (reaction conditions) of reactor object into a dictionary
    reactor_i = {}
    reactor_i['C0'] = C0
    reactor_i['tf'] = tf
    reactor_i['names'] = rxn_names
    reactor_i['temperature'] = temperature
    reactor_data.append(reactor_i)
    # # Use profile as the "experimental" data, n_reactor = 1
    Y_experiments.append(C_profile)
    # Set the weight for water as zero, others as 1

# Input experimental data and models (rate expressions) into a model bridge
bridge = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
bridge.input_data(stoichiometry, reactor_data, Y_experiments, Y_weights, t_eval, qoi='profile')

# # Test the bridge on ground truth parameters, the loss should be 0
# t_vec_predict, Y_predict = bridge.profile(para_ground_truth)
# loss_ground_truth = bridge.loss_func(para_ground_truth)
# for i, T in enumerate(Kp["EDH"].keys()):
#     plot_ethane_overlap(t_vec_predict[i], Y_predict[i], t_eval, Y_experiments[i], 'Ground truth @ T ' + str(T),
#                         estimator_name)

# Set up an optimizer
# Automatically compute the parameter ranges given a deviation
deviation = 0.25
para_ranges = [[-np.abs(vi) * deviation + vi, np.abs(vi) * deviation + vi] for vi in para_ground_truth]

# Start a timer
start_time = time.time()
n_iter = 300
optimizer = BOOptimizer(estimator_name)
X_opt, loss_opt, Exp = optimizer.optimize(bridge.loss_func, para_ranges, n_iter, log_flag=True)
end_time = time.time()

# Predict the conversions given the optimal set of parameters
t_opt, Y_opt = bridge.profile(X_opt)
for i, T in enumerate(Kp["EDH"].keys()):
    plot_ethane_overlap(t_opt[i], Y_opt[i], t_eval, Y_experiments[i], 'Optimal Set @ T=' + str(T), estimator_name)
plot_ethane_residual(t_opt[3], Y_opt[3], t_eval, Y_experiments[3], 'Residual @ T=873K', estimator_name)
# Print the results
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt, para_ground_truth)


# %% Test on the ethane system, with noise
# Add to noise to the concentrations
estimator_name = 'ethane_multiple_T_noisy'
ut.clear_cache(estimator_name)

reactor_data = []
Y_experiments_noisy = []
n_reactor = len(Kp["EDH"].keys())
Y_weights = np.ones(m_specs)
Y_weights[-1] = 0.
for T in Kp["EDH"].keys():
    temperature = float(T)
    rate_EDH = rate_eq(C0, para_ethane, stoichiometry[0], 'EDH', temperature)
    rate_Hyd = rate_eq(C0, para_ethane, stoichiometry[1], 'Hyd', temperature)
    rate_RWGS = rate_eq(C0, para_ethane, stoichiometry[2], 'RWGS', temperature)
    # Test on a reactor model to perform integration
    reactor_ethane = Reactor(stoichiometry, tf, C0=C0, names=rxn_names, temperature=temperature)
    tC_profile = reactor_ethane.get_profile(rate_eq, para_ethane, t_eval=t_eval)
    # Plot the profile
    t_eval = tC_profile[:, 0]
    C_profile = tC_profile[:, 1:]
    noise_level = 0.00001
    noise_matrix = np.random.normal(loc=0, scale=noise_level, size=C_profile.shape)
    C_profile_noisy = noise_matrix + C_profile
    plot_ethane_overlap(t_eval, C_profile, t_eval, C_profile+noise_matrix, 'Noisy data @T=' + str(T) + ' (K)', estimator_name)
    # Parse the specifics (reaction conditions) of reactor object into a dictionary
    reactor_i = {}
    reactor_i['C0'] = C0
    reactor_i['tf'] = tf
    reactor_i['names'] = rxn_names
    reactor_i['temperature'] = temperature
    reactor_data.append(reactor_i)
    # # Use profile as the "experimental" data, n_reactor = 1
    Y_experiments_noisy.append(C_profile_noisy)
    # Set the weight for water as zero, others as 1
# Construct a ModelBridge object
bridge_noisy = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
bridge_noisy.input_data(stoichiometry, reactor_data, Y_experiments_noisy, Y_weights, t_eval=t_eval, qoi='profile')

# Perform the optimization
start_time = time.time()
n_iter = 200
optimizer_noisy = BOOptimizer(estimator_name)
X_opt_noisy, loss_opt_noisy, Exp_noisy = optimizer_noisy.optimize(bridge_noisy.loss_func, para_ranges, n_iter,
                                                                  log_flag=True)
end_time = time.time()

# Predict the conversions given the optimal set of parameters
t_opt_noisy, Y_opt_noisy = bridge_noisy.profile(X_opt_noisy)
for i, T in enumerate(Kp["EDH"].keys()):
    plot_ethane_overlap(t_opt_noisy[i], Y_opt_noisy[i], t_eval, Y_experiments_noisy[i], 'Optimal Set @ T=' + str(T), estimator_name)
    plot_ethane_residual(t_opt_noisy[i], Y_opt_noisy[i], t_eval, Y_experiments_noisy[i], 'Residual Noisy @873K', estimator_name)

# Print the results
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt, para_ground_truth)

