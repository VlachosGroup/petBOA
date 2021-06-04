"""
Test on ethane_dehydrogenation system ODEs
"""
import pandas as pd
from estimator.reactor import Reactor
from examples.batch_reactor_template.ethane_dehydrogenation.fit_all_params.ethane_model_load import *

# Reaction equations:
# EDH: ethane_dehydrogenation dehydrogenation: C2H6 -> C2H4 + H2
# Hyd: hydrogenolysis: C2H6 + H2 -> 2CH4
# RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O

# Set ground truth parameter values
A0_EDH = 2.5E6
Ea_EDH = 125  # kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110  # kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70  # kJ/mol

# Kp values calculated at different T
# T kp value pair in a dictionary

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

# Inlet Concentration Values in M
C2H6_in = 0.0005
C2H4_in = 0
CH4_in = 0
H2_in = 0.0005
CO2_in = 0.0005
CO_in = 0
H2O_in = 0
C0 = [C2H6_in, C2H4_in, CH4_in, H2_in, CO2_in, CO_in, H2O_in]

# Integration time
tf = 10

# we can set the integration t_eval (optional)
n_int = 11
t_eval = np.linspace(0, tf, n_int)

# %% Tests on the ethane_dehydrogenation system at different temperatures
# Test on whether the rate can be calculated correctly
data = []
P = 1.00  # atm
for T in Kp["EDH"].keys():
    temperature = float(T)
    reactor_ethane_i = Reactor(stoichiometry, tf, C0=C0, names=rxn_names, temperature=temperature)
    tC_profile = reactor_ethane_i.get_profile(rate_eq, para_ethane, t_eval=t_eval)
    # Plot the profile
    t_eval = tC_profile[:, 0]
    C_profile = tC_profile[:, 1:]
    for i, time in enumerate(t_eval):
        _list = [float(T), P]
        for entry in C_profile[i]:
            _list.append(entry)
        data.append(_list)
run_names = ["run_" + str(i) for i in range(len(Kp["EDH"].keys()))]
ind_names = pd.MultiIndex.from_product([run_names, t_eval])
Data = np.array(data).reshape(4 * 11, 9)
col_names = ["Temperature(K)", "Pressure(atm)", "y_C2H6", "y_C2H4", "y_CH4", "y_H2", "y_CO2", "y_CO", "y_H2O"]
df = pd.DataFrame(Data, index=ind_names, columns=col_names)
df = df.rename_axis(index=("run_no", "time(s)"))
df.to_csv("ethane_batch_expt_data.csv")
