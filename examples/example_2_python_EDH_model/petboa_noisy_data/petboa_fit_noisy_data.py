"""
Fit parameters using the simulated data with added noise
for the Ethane dehydrogenation batch reactor
with the Reaction equations:

EDH: ethane dehydrogenation: C2H6 -> C2H4 + H2
Hyd: hydrogenolysis: C2H6 + H2 -> 2CH4
RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O
"""
import os
import time

import pandas as pd
import petboa.utils as ut
from petboa.optimizer import BOOptimizer
from petboa.plots import plot_overlap
from petboa.reactor import ModelBridge
from petboa.utils import WeightedRMSE, RMSE, parse_param_file
import numpy as np
from ethane_model import stoichiometry, rxn_names, rate_eq, Kp, m_specs


# # Write the loss function
def loss_func(self, x):
    """Generic loss function"""
    loss = 0
    y_predict = self.profile(x, return_t_eval=False)
    for _ in range(wrapper.n_reactors):
        if self.Y_weights is None:
            loss += RMSE(self.Y_groundtruth[_]*1E3, y_predict[_]*1E3)
        else:  # Factor in the weights
            loss += WeightedRMSE(self.Y_groundtruth[_]*1E3, y_predict[_]*1E3, self.Y_weights)
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss])
    self.param_evolution.append([self.call_count] + list(x))
    print("In iteration {} Loss is {:.5f} parameters are {:.2f} {:.2f} {:.2f}"
          "{:.2f} {:.2f} {:.2f}".format(self.call_count, loss, *x))
    return loss


# Start the parameter estimation problem and set an estimator name
estimator_name = 'fit_noisy_params'
ut.clear_cache(estimator_name)
basedir = os.getcwd()
# Load the experimental data from the csv file.
new_data = pd.read_csv('../gen_data/noisy_expt_data.csv')
new_data = new_data.set_index(['run_no', 'time(s)'])
temperatures = new_data['Temperature(K)'].unique()
expt_runs = new_data.reset_index()['run_no'].unique()
reactor_run_data = []
Y_experiments = []


Y_weights = np.ones(m_specs)
# Optionally set the weight for water as zero, others as 1
Y_weights[-1] = 0.

tf = 10.0
# we can set the integration t_eval (optional)
n_int = 11
t_eval = np.linspace(0, tf, n_int)
for run in expt_runs[:]:
    _list_0 = new_data.loc[run, 0.0]  # data from the csv file corresponding to time t=0
    reactor_i = {'C0': _list_0.to_numpy()[2:], 'tf': tf,
                 'temperature': _list_0['Temperature(K)'], 'names': rxn_names}
    _list_1 = []
    for d in new_data.loc[run, :].to_numpy():
        _list_1.append(d[2:])
    conc_profile = _list_1  # concentration profile
    reactor_run_data.append(reactor_i)
    # Use concentration profile as the "experimental" data,
    Y_experiment_i = np.array(conc_profile)
    Y_experiments.append(Y_experiment_i)

#  Define the kinetic parameters
#  6 parameters to fit
#  Ground Truth of the parameters
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

para_name_ethane, para_ranges = parse_param_file("params.xlsx")

# Alternatively set it up manually
# para_name_ethane = list(para_ethane.keys())
# para_ranges = [[6, 9],  # EDH_prefactor
#                [50, 200],  # EDH_Ea
#                [6, 9],  # Hyd_prefactor
#                [50, 200],  # Hyd_Ea
#                [6, 9],  # RWGS_prefactor
#                [50, 200],  # RWGS_Ea
#                ]


if not os.path.exists(estimator_name):
    os.mkdir(estimator_name)

a = open('output.log', mode='w')
print("Parameter Names {} {} {} {} {} {}".format(*para_name_ethane))
print("Ground Truth {} {} {} {} {} {}".format(*para_ethane.values()))
a.write("Parameter Names {} {} {} {} {} {} \n".format(*para_name_ethane))
a.write("Ground Truth {} {} {} {} {} {} \n".format(*para_ethane.values()))

param_res = []
full_df = pd.DataFrame()
for repeat in range(5):
    # Input experimental data and models (rate expressions) into a model wrapper
    ModelBridge.loss_func = loss_func
    wrapper = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
    wrapper.input_data(stoichiometry, reactor_run_data,
                       Y_experiments, Y_weights, t_eval, qoi='profile')
    start_time = time.time()
    print("###################### \n")
    print("Repeat {} \n".format(str(repeat)))
    print("###################### \n")
    a.write("###################### \n")
    a.write("Repeat {} \n".format(str(repeat)))
    a.write("###################### \n")
    # # Set up an optimizer
    # # Start a timer
    optimizer = BOOptimizer(estimator_name)
    n_iter = 140
    X_opt, loss_opt, Exp = optimizer.optimize(objective_func=wrapper.loss_func,
                                              para_ranges=para_ranges,
                                              acq_func='EI',
                                              n_iter=n_iter,
                                              n_sample_multiplier=10,
                                              log_flag=True)
    end_time = time.time()
    ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)
    a.write("Objective function called {} times \n".format(wrapper.call_count))
    a.write("Parameters are {} \n".format(X_opt))
    a.write("Total Time: {} sec \n".format(end_time - start_time))
    df1 = pd.DataFrame(data=wrapper.loss_evolution,
                       columns=['Run No', 'Loss'],
                       )
    df1['Repeat'] = repeat
    df2 = pd.DataFrame(data=wrapper.param_evolution,
                       columns=['Run No'] + para_name_ethane,
                       )
    df1 = df1.merge(df2, how='inner', on='Run No')
    full_df = pd.concat([full_df, df1])
    param_res.append(X_opt)

os.chdir(estimator_name)
full_df.to_csv("param_loss_history.csv")
df3 = pd.DataFrame(data=param_res,
                   columns=para_name_ethane,
                   )
df3 = df3.T
df3['mean'] = df3.mean(axis=1)
df3['std'] = df3.std(axis=1)
df3.to_csv("final_fit_parameters.csv")
print(df3)

# Plot the profiles given the optimal set of parameters
os.chdir(basedir)
wrapper = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
wrapper.input_data(stoichiometry, reactor_run_data,
                   Y_experiments, Y_weights, t_eval, qoi='profile')
X_opt = df3['mean'].values
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
                 fig_name='T= {} K'.format(T),
                 legend_labels=legend_labels,
                 save_path=estimator_name)
