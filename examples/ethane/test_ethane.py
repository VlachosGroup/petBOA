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


def plot_ethane(t_vec, C_profile, title=None):
    """
    plot the ode profiles 
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(t_vec, C_profile[:,0], label = r'$\rm C_{2}H_{6}$')
    ax.plot(t_vec, C_profile[:,1], label = r'$\rm C_{2}H_{4}$')
    ax.plot(t_vec, C_profile[:,2], label = r'$\rm CH_{4}$')
    ax.plot(t_vec, C_profile[:,3], label = r'$\rm H_{2}$')
    ax.plot(t_vec, C_profile[:,4], label = r'$\rm CO_{2}$')
    ax.plot(t_vec, C_profile[:,5], label = r'$\rm CO$')
    ax.plot(t_vec, C_profile[:,6], label = r'$\rm H_{2}O$')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('C (mol/L)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title is not None:
        ax.set_title(title)


def plot_ethane_overlap(t_vec1, C_profile1, t_vec2, C_profile2, title=None):
    """
    Plot the ode profiles, 
    Check whether the first profile matches with the second
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(t_vec1, C_profile1[:,0], label = r'$\rm C_{2}H_{6}$')
    ax.plot(t_vec1, C_profile1[:,1], label = r'$\rm C_{2}H_{4}$')
    ax.plot(t_vec1, C_profile1[:,2], label = r'$\rm CH_{4}$')
    ax.plot(t_vec1, C_profile1[:,3], label = r'$\rm H_{2}$')
    ax.plot(t_vec1, C_profile1[:,4], label = r'$\rm CO_{2}$')
    ax.plot(t_vec1, C_profile1[:,5], label = r'$\rm CO$')
    ax.plot(t_vec1, C_profile1[:,6], label = r'$\rm H_{2}O$')
    # scatter for the second profile
    ax.scatter(t_vec2, C_profile2[:,0], s= 5, alpha = 0.3)
    ax.scatter(t_vec2, C_profile2[:,1], s= 5, alpha = 0.3)
    ax.scatter(t_vec2, C_profile2[:,2], s= 5, alpha = 0.3)
    ax.scatter(t_vec2, C_profile2[:,3], s= 5, alpha = 0.3)
    ax.scatter(t_vec2, C_profile2[:,4], s= 5, alpha = 0.3)
    ax.scatter(t_vec2, C_profile2[:,5], s= 5, alpha = 0.3)
    ax.scatter(t_vec2, C_profile2[:,6], s= 5, alpha = 0.3)
    
    ax.set_xlabel('t (s)')
    ax.set_ylabel('C (mol/L)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title is not None:
        ax.set_title(title)
        
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

#%% Tests on the ethane system, with no noise
# Test on whether the rate can be calculated correctly
rate_EDH = rate_eq(C0, para_ethane, stoichiometry[0],  'EDH', temperature)
rate_Hyd = rate_eq(C0, para_ethane, stoichiometry[1], 'Hyd', temperature)
rate_RWGS = rate_eq(C0, para_ethane, stoichiometry[2],'RWGS',  temperature)


# Test on a reactor model to perform integration 
reactor_ethane = Reactor(stoichiometry, tf, C0=C0, names=rxn_names, temperature=temperature)
tC_profile = reactor_ethane.get_profile(rate_eq, para_ethane, t_eval=t_eval)
# Plot the profile
t_eval = tC_profile[:, 0]
C_profile = tC_profile[:, 1:]
plot_ethane(t_eval, C_profile, 'Ground truth')


# Test the model bridge
# Prase the specifics (reaction conditions) of reactor object into a dictionary
reactor_i = {}
reactor_i['C0'] = C0
reactor_i['tf'] = tf
reactor_i['names'] = rxn_names
reactor_i['temperature'] = temperature
reactor_data = [reactor_i]

# Use profile as the "experimental" data, n_reactor = 1
n_reactor = 1
Y_experiments = [C_profile]

# Set the weight for water as zero, others as 1 
Y_weights = np.ones((n_reactor,m_specs))
Y_weights[:,-1] = 0 

# Input experimental data and models (rate expressions) into a model bridge
bridge = ModelBridge(rate_eq, para_name_ethane, name = 'rate_ethane')
bridge.input_data(stoichiometry, reactor_data, Y_experiments, Y_weights, t_eval=t_eval, eval_profile=True)

# Test the bridge on a set of 1s
para_set_1 = np.ones(len(para_name_ethane))
t_vec_predict, Y_predict = bridge.profile(para_set_1)
loss_1 = bridge.loss_func(para_set_1)
plot_ethane(t_vec_predict[0], Y_predict[0], 'Set of 1s')

# Test the bridge on ground truth parameters, the loss should be 0
t_vec_predict, Y_predict = bridge.profile(para_ground_truth)
loss_ground_truth = bridge.loss_func(para_ground_truth)
plot_ethane_overlap(t_vec_predict[0], Y_predict[0], t_eval, C_profile, 'Ground truth')


# Set up an optimizer 
# Automatically compute the parameter ranges given a deviation
deviation = 0.5
para_ranges =  [[-np.abs(vi) *deviation + vi, np.abs(vi) *deviation + vi] for vi in para_ethane.values()]

# Start a timer
start_time = time.time()
n_iter = 100
optimizer = BOOptimizer()
X_opt, loss_opt, Exp = optimizer.optimize(bridge.loss_func, para_ranges, n_iter, log_flag=True)
end_time= time.time()

# Predict the conversions given the optimal set of parameters
t_opt, Y_opt = bridge.profile(X_opt)
plot_ethane_overlap(t_opt[0], Y_opt[0], t_eval, C_profile,'Optimal Set')

# Check the left and right bounds without optimization
para_left = [vi[0] for vi in para_ranges]
para_right = [vi[1] for vi in para_ranges]

t_left, Y_left = bridge.profile(para_left)
loss_left = bridge.loss_func(para_left)
plot_ethane_overlap(t_left[0], Y_left[0],t_eval, C_profile, 'Left bound')

t_right, Y_right = bridge.profile(para_right)
loss_right = bridge.loss_func(para_right)
plot_ethane_overlap(t_right[0], Y_right[0], t_eval, C_profile,'Right bound')


# Print the results
file = open(bridge.name + ".txt","w")
file.write('Parameter estimation takes {:.2f} min \n'.format((end_time-start_time)/60))
file.write('Final loss {:.6f} \n'.format(loss_opt))
file.write('Parameters are {} \n'.format(X_opt))
file.close()


#%% Test on the ethane system, with noise
# Add to noise to the concentrations
noise_level = 0.00003
noise_matrix = np.random.normal(loc=0,scale=noise_level, size=C_profile.shape)
C_profile_noisy = noise_matrix + C_profile
plot_ethane_overlap(t_eval, C_profile_noisy, t_eval, C_profile, 'Noisy Data')

# Update the experimental data
Y_experiments_noisy = [C_profile_noisy]

# Construct a ModelBridge object
bridge_noisy = ModelBridge(rate_eq, para_name_ethane, name = 'rate_ethane_noisy')
bridge_noisy.input_data(stoichiometry, reactor_data, Y_experiments_noisy, Y_weights, t_eval=t_eval, eval_profile=True)

# Perform the optimization
start_time = time.time()
n_iter = 100
optimizer_noisy = BOOptimizer()
X_opt_noisy, loss_opt_noisy, Exp_noisy = optimizer_noisy.optimize(bridge_noisy.loss_func, para_ranges, n_iter, log_flag=True)
end_time= time.time()


# Predict the conversions given the optimal set of parameters
t_opt_noisy, Y_opt_noisy = bridge_noisy.profile(X_opt_noisy)
plot_ethane_overlap(t_opt_noisy[0], Y_opt_noisy[0], t_eval, C_profile, 'Optimal Set for noisy data')


# Print the results
file = open(bridge_noisy.name + ".txt","w")
file.write('Parameter estimation takes {:.2f} min \n'.format((end_time-start_time)/60))
file.write('Final loss {:.6f} \n'.format(loss_opt_noisy))
file.write('Parameters are {} \n'.format(X_opt_noisy))
file.close()