"""
Estimate the Global sensitivity coefficients
for the ethane_dehydrogenation system ODEs
"""
import os

import numpy as np
import pandas as pd
import petboa.utils as ut
from SALib.analyze.sobol import analyze
from SALib.sample import sobol
from petboa.reactor import ModelBridge
from petboa.utils import WeightedRMSE, RMSE

from ethane_model import stoichiometry, rxn_names, m_specs, rate_eq


# # Write the loss function
def loss_func(self, x):
    """Generic loss function"""
    loss = 0
    y_predict = self.profile(x, return_t_eval=False)
    for _ in range(wrapper.n_reactors):
        if self.Y_weights is None:
            loss += RMSE(self.Y_groundtruth[_], y_predict[_])
        else:  # Factor in the weights
            loss += WeightedRMSE(self.Y_groundtruth[_], y_predict[_], self.Y_weights)
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss])
    self.param_evolution.append([self.call_count] + list(x))
    print("In iteration {} Loss is {:.5f} parameters are {:.2f} {:.2f} {:.2f}"
          "{:.2f} {:.2f} {:.2f}".format(self.call_count, loss, *x))
    return loss


# Start the parameter estimation problem and set an estimator name
estimator_name = 'outputs-gsa-salib'
ut.clear_cache(estimator_name)

# Load the experimental data from the csv file.
new_data = pd.read_csv('../gen_data/expt_data.csv')
new_data = new_data.set_index(['run_no', 'time(s)'])
temperatures = new_data['Temperature(K)'].unique()
expt_runs = new_data.reset_index()['run_no'].unique()
reactor_run_data = []
Y_experiments = []
Y_weights = np.ones(m_specs)
Y_weights[-1] = 0.  # Set the weight for wate r as zero, others as 1
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

para_name_ethane = list(para_ethane.keys())
if not os.path.exists(estimator_name):
    os.mkdir(estimator_name)
os.chdir(estimator_name)
a = open('output.log', mode='w')
print("Parameter Names {} {} {} {} {} {}".format(*para_name_ethane))
print("Ground Truth {} {} {} {} {} {}".format(*para_ethane.values()))
a.write("Parameter Names {} {} {} {} {} {} \n".format(*para_name_ethane))
a.write("Ground Truth {} {} {} {} {} {} \n".format(*para_ethane.values()))

para_ranges = [[6, 9],  # EDH_prefactor
               [50, 200],  # EDH_Ea
               [6, 9],  # Hyd_prefactor
               [50, 200],  # Hyd_Ea
               [6, 9],  # RWGS_prefactor
               [50, 200],  # RWGS_Ea
               ]

ModelBridge.loss_func = loss_func
wrapper = ModelBridge(rate_eq, para_name_ethane, name=estimator_name)
wrapper.input_data(stoichiometry, reactor_run_data,
                   Y_experiments, Y_weights, t_eval, qoi='profile')

problem = {
    'num_vars': len(para_name_ethane),
    'names': para_name_ethane,
    'bounds': para_ranges
}
param_values = sobol.sample(problem, 128)
print("Total number of sensitivity run samples {}".format(param_values.shape[0]))
a.write("Total number of sensitivity run samples {} \n".format(param_values.shape[0]))
Y = np.zeros([param_values.shape[0]])
for i, X in enumerate(param_values):
    Y[i] = wrapper.loss_func(X)
    a.write("Finished evaluating {} Sobol sample {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}"
            "{:.2f} with loss {:.5f} \n".format(i, *X, Y[i]))

history = np.append(param_values, np.reshape(Y, (len(Y), 1)), axis=1)
hist_df = pd.DataFrame(data=history,
                       columns=para_name_ethane+["loss"])
hist_df.to_csv("gsa_sobol_history.csv")
si = analyze(problem, Y)
a.write("\n \nFirst Order Sensitivity Indices are:\n {} \n".format(si['S1']))
a.write("Total Order Sensitivity Indices are:\n {} \n".format(si['ST']))

data = {"ST": pd.Series(data=si['ST'],
                        index=para_name_ethane),
        "S1": pd.Series(data=si['S1'],
                        index=para_name_ethane)}

df = pd.DataFrame(data=data)
print(df)
df.to_csv('gsa-results.csv')
