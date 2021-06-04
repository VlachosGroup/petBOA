"""
Test on ethane_dehydrogenation system ODEs
"""
import os
import time
import sys
import estimator.utils as ut
from estimator.plots import plot_overlap, plot_residual
from estimator.optimizer import BOOptimizer
from estimator.reactor import ModelBridge
import pandas as pd
from examples.batch_reactor_template.ethane_dehydrogenation.fit_all_params_multiple_T.ethane_model_load import *

# Start the parameter estimation problem and set an estimator name
# Estimator name
estimator_name = 'ethane_batch_results'
ut.clear_cache(estimator_name)

# Load the experimental data from the csv file.
new_data = pd.read_csv('ethane_batch_expt_data.csv')
new_data = new_data.set_index(['run_no', 'time(s)'])
temperatures = new_data['Temperature(K)'].unique()
expt_runs = new_data.reset_index()['run_no'].unique()
reactor_run_data = []
Y_experiments = []
Y_weights = np.ones(m_specs)
Y_weights[-1] = 0.  # Set the weight for water as zero, others as 1
tf = 10
# we can set the integration t_eval (optional)
n_int = 11
t_eval = np.linspace(0, tf, n_int)
for run in expt_runs[:]:
    _list_0 = new_data.loc[run, 0.0]  # data from the csv file corresponding to time t=0
    reactor_i = {'C0': _list_0.to_numpy()[2:], 'tf': tf,
                 'temperature': _list_0['Temperature(K)'], 'names': rxn_names}
    _list_1 = []
    for d in new_data.index:
        if run in d:
            _list_1.append(new_data.loc[d].to_numpy()[2:])
    conc_profile = _list_1  # concentration profile
    reactor_run_data.append(reactor_i)
    # Use concentration profile as the "experimental" data,
    Y_experiment_i = np.array(conc_profile)
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
                   Y_experiments, Y_weights, t_eval, qoi='profile')

# # Set up an optimizer
# # Start a timer
start_time = time.time()
n_iter = 100
optimizer = BOOptimizer(estimator_name)
X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func, para_ranges, n_iter, log_flag=True)
end_time = time.time()

# Predict the conversions given the optimal set of parameters
t_opt, Y_opt = wrapper.profile(X_opt)
legend_labels = [r'$\rm C_{2}H_{6}$',
                 r'$\rm C_{2}H_{4}$',
                 r'$\rm CH_{4}$',
                 r'$\rm H_{2}$',
                 r'$\rm CO_{2}$',
                 r'$\rm CO$',
                 r'$\rm H_{2}O$']
for i, T in enumerate(Kp["EDH"].keys()):
    plot_overlap(t_opt[i], Y_opt[i], t_eval, Y_experiments[i],
                 title='Optimal Set @ T=' + str(T),
                 legend_labels=legend_labels,
                 save_path=estimator_name)
print(t_opt[3], Y_opt[3] - Y_experiments[3])
print(t_opt[3], np.array(Y_opt[3] - Y_experiments[3]).T)
plot_residual(t_opt[3], Y_opt[3], t_eval, Y_experiments[3],
              title='Residual @ T=873K',
              legend_labels=legend_labels,
              save_path=estimator_name)
# Print the results
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)
