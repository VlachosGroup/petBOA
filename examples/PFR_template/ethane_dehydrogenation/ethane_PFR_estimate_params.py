"""
Parameter estimation for ethane dehydrogenation PFR reactor
"""
import time

import matplotlib.pyplot as plt

import estimator.utils as ut
from estimator.optimizer import BOOptimizer
from estimator.reactor import Reactor, ModelBridge
from examples.PFR_template.ethane_dehydrogenation.ethane_model_load import *

# Reaction equations:
# EDH: ethane dehydrogenation: C2H6 -> C2H4 + H2
# Hyd: hydrogenolysis: C2H6 + H2 -> 2CH4
# RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O

# Start the parameter estimation problem and set an estimator name
estimator_name = 'ethane_PFR_results'
ut.clear_cache(estimator_name)

# Load the experimental data from the csv file.
new_data = pd.read_csv('ethane_pfr_data.csv')
new_data = new_data.set_index(['run_no', 'time/tau'])
temperatures = new_data['Temperature(K)'].unique()
expt_runs = new_data.reset_index()['run_no'].unique()
reactor_run_data = []
Y_experiments = []
Y_weights = np.ones(m_specs)
Y_weights[-1] = 0.  # Set the weight for water as zero, others as 1
for run in expt_runs:
    _list_0 = new_data.loc[run, 0]  # data from the csv file corresponding to time t=0
    _list_1 = new_data.loc[run, 1]  # data from the csv file corresponding to time t=tau
    reactor_i = {'C0': _list_0.to_numpy()[3:], 'tf': _list_0['Residence time(s)'],
                 'temperature': _list_0['Temperature(K)'], 'names': rxn_names}
    exit_concentration = _list_1.to_numpy()[3:]  # exit concentration
    reactor_run_data.append(reactor_i)
    # Use exit concentrations as the "experimental" data, n_reactor = nT * n_tf
    Y_experiment_i = exit_concentration
    Y_experiments.append(Y_experiment_i)

#  Define the kinetic parameters
#  6 parameters to fit
#  initial guesses for the parameters
A0_EDH = 2.5E6
Ea_EDH = 125  # kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110  # kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70  # kJ/mol
para_ethane = {'EDH_prefactor': np.log10(A0_EDH),
               'EDH_Ea': Ea_EDH,
               'Hyd_prefactor': np.log10(A0_Hyd),
               'Hyd_Ea': Ea_Hyd,
               'RWGS_prefactor': np.log10(A0_RWGS),
               'RWGS_Ea': Ea_RWGS
               }
para_name_ethane = list(para_ethane.keys())

# parameter bounds that are used for parameter estimation
deviation = 0.25
para_ranges = [[-np.abs(vi) * deviation + vi, np.abs(vi) * deviation + vi] for vi in para_ethane.values()]
# #%% Input experimental data and models (rate expressions) into a model wrapper
wrapper = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
wrapper.input_data(stoichiometry, reactor_run_data,
                   Y_experiments, Y_weights, qoi='concentration')  # loss function is based on exit concentration

# %% Set up an optimizer
#  Start a timer
print("Starting parameter estimation")
start_time = time.time()
n_iter = 50
optimizer = BOOptimizer(estimator_name)
X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func,
                                          para_ranges,
                                          n_iter=n_iter,
                                          n_sample_multiplier=3,
                                          log_flag=True)
end_time = time.time()
print("Loss Function called {} ".format(wrapper.call_count))
print("Model Function called {} ".format(wrapper.model_count))
# #%% Compare the estimated profiles to the ground truth data points

# Optimal parameter values
para_ethane_opt = ut.para_values_to_dict(X_opt, para_name_ethane)
# lists of estimated profile at each temperature
t_opt = []
Y_opt = []
Y_data = []
n_int = 11
# Predict the conversions given the optimal set of parameters
for i, run in enumerate(expt_runs):
    _list_0 = new_data.loc[run, 0]  # data from the csv file corresponding to time t=0
    _list_1 = new_data.loc[run, 1]  # data from the csv file corresponding to time t=tau
    C0 = _list_0.to_numpy()[3:]
    tf = _list_0['Residence time(s)']
    T = _list_0['Temperature(K)']
    reactor_ethane = Reactor(stoichiometry, tf, C0=C0, names=rxn_names, temperature=T)
    tC_profile = reactor_ethane.get_profile(rate_eq, para_ethane_opt, t_eval=np.linspace(0, tf, n_int))
    Y_opt_i = tC_profile[-1, 1:]
    Y_opt.append(Y_opt_i)
    Y_data.append(_list_1.to_numpy()[3:])

# Set matplotlib default values
font = {'size': 20}
import matplotlib

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
legend_labels = [r'$\rm C_{2}H_{6}$',
                 r'$\rm C_{2}H_{4}$',
                 r'$\rm CH_{4}$',
                 r'$\rm H_{2}$',
                 r'$\rm CO_{2}$',
                 r'$\rm CO$',
                 r'$\rm H_{2}O$']

fig, ax = plt.subplots(2, 4, figsize=(20, 10))
axs = [ax[i, j] for i in range(2) for j in range(4)]
for i, item in enumerate(np.array(Y_opt).T):
    axs[i].scatter(y=np.array(Y_opt).T[i],
                   x=np.array(Y_data).T[i], )
    axs[i].plot(np.array(Y_data).T[i],
                np.array(Y_data).T[i],
                color='black')
    axs[i].set_xlabel(xlabel=legend_labels[i] + "-obs (mol/l)")
    axs[i].set_ylabel(ylabel=legend_labels[i] + "-pred (mol/l)")
plt.tight_layout()
plt.savefig(estimator_name + '/Parity-Plot-Ethane-PFR')

# %%
# Print the results
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)