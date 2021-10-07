# coding: utf-8
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmutt.constants
from pmutt import pmutt_list_to_dict
from pmutt.empirical.nasa import Nasa
from pmutt.empirical.references import Reference, References
from pmutt.io.excel import read_excel
from pmutt.io.omkm import organize_phases, write_cti
from pmutt.mixture.cov import PiecewiseCovEffect
from pmutt.omkm.reaction import BEP, SurfaceReaction
from pmutt.omkm.units import Units
from pathlib import Path

import estimator.utils as ut
from estimator.modelwrappers import ModelWrapper
from estimator.omkm import OMKM
from estimator.optimizer import BOOptimizer


def edit_yaml(filename, p, t, q):
    f = open(filename, 'w')
    f.write("""# File generated by pMuTT (v 1.2.21) on 2020-07-09 16:03:04.988190
# See documentation for OpenMKM YAML file here:
# https://vlachosgroup.github.io/openmkm/input
inlet_gas:
    flow_rate: \"""" + str(q) + """ cm3/s"
phases:
    bulk:
        name: bulk
    gas:
        initial_state: "NH3:1.0"
        name: gas
    surfaces:
    -   initial_state: "RU(T):1.0"
        name: terrace
    -   initial_state: "RU(S):1.0"
        name: step
reactor:
    cat_abyv: "1500 /cm"
    temperature_mode: "isothermal"
    pressure_mode: "isobaric"
    pressure: \"""" + str(p) + """ atm"
    temperature: """ + str(t) + """
    type: "cstr"
    volume: "1.0 cm3"
simulation:
    end_time: "50 s"
    init_step: 1.0e-15
    output_format: "csv"
    solver:
        atol: 1.0e-15
        rtol: 1.0e-10
    stepping: "logarithmic"
    transient: false""")
    f.close()


def make_thermo_input(input_path, params, param_keys=None):
    R = pmutt.constants.R(units='kJ/mol/K')
    # First, we will designate the units to write the CTI and YAML file.
    units_data = read_excel(io=input_path, sheet_name='units')[0]
    units = Units(**units_data)

    # Second, we will open the input spreadsheet and read the `refs` sheet.
    try:
        refs_data = read_excel(io=input_path, sheet_name='refs')
    except:
        # If references are not used, skip this section
        print(('The "refs" sheet could not be found in {}.'
               'Skipping references'.format(input_path)))
        refs = None
    else:
        refs = [Reference(**ref_data) for ref_data in refs_data]
        refs = References(references=refs)

    # Third, we will use the ``refs`` defined before and the ``species`` sheet to convert statistical mechanical data to
    # (https://vlachosgroup.github.io/pMuTT/api/empirical/nasa/pmutt.empirical.nasa.Nasa.html#pmutt.empirical.nasa.Nasa).
    # Read the species' data
    species_data = read_excel(io=input_path, sheet_name='species')

    # Create NASA polynomials from the species
    species = [Nasa.from_model(references=refs, **ind_species_data) \
               for ind_species_data in species_data]

    # perturb species energies
    if param_keys is not None:
        for i, key in enumerate(param_keys):
            for spec in species:
                if spec.name == key:
                    spec.a_high[-2] += params[i] / R
                    spec.a_low[-2] += params[i] / R
    else:
        # use alternate logic
        print("Param keys not associated with param values")
    # this is the core code

    # Reading BEP (optional)
    # Next, we read the BEP relationships to include.
    try:
        beps_data = read_excel(io=input_path, sheet_name='beps')
    except:
        print(('The "beps" sheet could not be found in {}. '
               'Skipping BEPs'.format(input_path)))
        beps = None
        species_with_beps = species.copy()
    else:
        beps = [BEP(**bep_data) for bep_data in beps_data]
        species_with_beps = species + beps

    # Read reactions
    # Convert species to dictionary for easier reaction assignment
    species_with_beps_dict = pmutt_list_to_dict(species_with_beps)

    reactions_data = read_excel(io=input_path, sheet_name='reactions')
    reactions = [SurfaceReaction.from_string(species=species_with_beps_dict, **reaction_data) \
                 for reaction_data in reactions_data]

    # Read lateral interactions (optional)
    # After, we read lateral interactions to include.
    try:
        interactions_data = read_excel(io=input_path,
                                       sheet_name='lateral_interactions')
    except:
        # If no lateral interactions exist, skip this section
        print(('The "lateral_interactions" sheet could not be found in {}.'
               'Skipping lateral interactions'.format(input_path)))
        interactions = None
    else:
        interactions = [PiecewiseCovEffect(**interaction_data) \
                        for interaction_data in interactions_data]

    # Reading Phases
    # Finally, we read the phases data from Excel and organize it for use in OpenMKM.
    # Read data from Excel sheet about phases
    phases_data = read_excel(io=input_path, sheet_name='phases')
    phases = organize_phases(phases_data, species=species, reactions=reactions,
                             interactions=interactions)
    reactor_data = read_excel(io=input_path, sheet_name='reactor')[0]

    # Write CTI File
    # The CTI file species the thermodynamics and kinetics of the system.
    # Note: We take the reactor operating conditions from YAML file to calculate thermodynamic and kinetic parameters
    cti_path = 'thermo_modified.cti'
    use_motz_wise = True
    T = reactor_data['T']
    write_cti(reactions=reactions, species=species, phases=phases, units=units,
              lateral_interactions=interactions, filename=cti_path,
              use_motz_wise=use_motz_wise, T=T, P=1.)


def loss_func(self,
              params=None,
              **kwargs):
    """
    Customized loss function specific to this problem
    """
    loss = 0.0
    reg_loss = 0.0
    # lamda is L2 regularization error multiplier
    if 'lamda' in kwargs.keys():
        lamda = kwargs['lamda']
    else:
        lamda = 0.0
    # alpha is RMSE error term multiplier
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 1.0
    os.chdir(self.model.wd_path)
    thermo_path = Path('inputs/NH3_Input_Data.xlsx').resolve()
    make_thermo_input(thermo_path,
                      params,
                      param_keys=self.para_names)
    for i in range(self.n_trials):
        _error = 0.0
        _reg_error = 0.0
        P = self.x_inputs[i][0]
        T = self.x_inputs[i][1]
        Q = self.x_inputs[i][2]
        y_predict = self.y_groundtruth[i][:]
        os.chdir(self.model.wd_path)
        edit_yaml(filename="reactor.yaml", p=P, t=T, q=Q)
        self.model.run(i)
        y_model = pd.read_csv("gas_mass_ss.csv").iloc[1][['N2', 'NH3', 'H2']].to_numpy()
        print(y_model, y_predict)
        _error = alpha * np.sqrt(np.mean((y_model - y_predict) ** 2))
        _reg_error = lamda * np.sqrt(np.mean((params) ** 2))
        reg_loss += _reg_error
        loss += _error + _reg_error
        print("Error {} Reg. Error {}".format(_error, _reg_error))
    #  end customization specific to the problem
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss, reg_loss])
    self.param_evolution.append(np.array(params))
    print("Optimization Iteration {} loss {}".format(self.call_count, loss))
    return loss


def get_parity(wrapper,
               params=None,
               **kwargs):
    os.chdir(wrapper.model.wd_path)
    thermo_path = Path('inputs/NH3_Input_Data.xlsx').resolve()
    make_thermo_input(thermo_path,
                      params,
                      param_keys=wrapper.para_names)
    y_data = []
    y_predict = []
    x_data = []
    loss = 0.0
    for i in range(wrapper.n_trials):
        P = wrapper.x_inputs[i][0]
        T = wrapper.x_inputs[i][1]
        Q = wrapper.x_inputs[i][2]
        y_exp = wrapper.y_groundtruth[i][:]
        os.chdir(wrapper.model.wd_path)
        edit_yaml(filename="reactor.yaml", p=P, t=T, q=Q)
        wrapper.model.run(i)
        y_model = pd.read_csv("gas_mass_ss.csv").iloc[1][['N2', 'NH3', 'H2']].to_numpy()
        print(y_exp, y_model)
        y_data.append(y_exp)
        y_predict.append(y_model)
        x_data.append([i, P, T, Q])
        _error = np.sqrt(np.mean((y_model - y_predict) ** 2))
        loss += _error
        print("Error {}".format(_error))
    print("Optimum params {} total loss {}".format(params, loss))
    os.chdir(wrapper.model.wd_path)
    return x_data, y_data, y_predict


def main():
    if os.name == 'nt':
        # provide windows path for the openmkm executable
        #example below
        omkm_path = "C:\\Users\\skasiraj\\source\\repos\\openmkm-VS\\x64\\Debug\\openmkm-VS.exe"
        wd_path = os.path.dirname(__file__)
    else:
        omkm_path = "omkm"
        wd_path = os.getcwd()
    omkm_instance = OMKM(exe_path=omkm_path,
                         wd_path=wd_path,
                         save_folders=False,
                         )
    a = open('debug.log', mode='w')
    data = pd.read_csv(filepath_or_buffer="all_data.csv")
    data.rename(columns={"Unnamed: 0": "exp_no"}, inplace=True)
    data.pop("t(s)")
    data = data[:]
    x_input = data[['pressure(atm)', 'temperature(K)', 'vol_flow_rate(cm3/sec)']].to_numpy()
    y_response = data[['N2_massfrac', 'NH3_massfrac', 'H2_massfrac']].to_numpy()

    # these are the bounds within which # parameters are searched
    # In our problem this is the enthalpy (perturbation/correction) of key species
    # we set it to be + or - 20 kJ/mol
    # parameter_range = [[-20.0, 20.0]] * 6
    parameter_range = [6.959, -4.629, -16.88, [-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]]
    para_ground_truth = {'N2(T)': 6.959765516670089,
                         'N(T)': -4.629843081311661,
                         'TS4_N2(T)': -16.88979206205836,
                         'N2(S)': 5.97427204174473,
                         'N(S)': 17.524267700455674,
                         'TS4_N2(S)': -10.841388069480615}
    start_time = time.time()
    estimator_name = 'outputs-omkm'
    ModelWrapper.loss_func = loss_func  # Connect loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=omkm_instance,  # openmkm wrapper with the "run" method
                           para_names=list(para_ground_truth.keys()),
                           name=estimator_name,
                           )
    wrapper.input_data(x_inputs=x_input,
                       n_trials=len(data),
                       y_groundtruth=y_response)

    optimizer = BOOptimizer(estimator_name)
    n_iter = 100
    X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func,
                                              parameter_range,
                                              n_sample_multiplier=20,
                                              n_iter=n_iter,
                                              log_flag=True,
                                              lamda=0.001)
    a.write("Objective function called {} times \t".format(wrapper.call_count))
    a.write("Parameters are {} \n".format(X_opt))
    end_time = time.time()
    a.write("Total Time: {} sec \n".format(end_time - start_time))
    os.chdir(wd_path)
    ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)

    df1 = pd.DataFrame(data=wrapper.loss_evolution,
                       columns=['Run No', 'Loss', 'Reg Loss'],
                       )
    df1.to_csv("{0}\\loss_history.csv".format(estimator_name))
    df2 = pd.DataFrame(data=wrapper.param_evolution,
                       columns=para_ground_truth.keys(),
                       )
    df2.to_csv("{0}\\param_history.csv".format(estimator_name))

    # Plot the results and parity plots
    x_data, y_data, y_opt = get_parity(wrapper, X_opt)
    legend_labels = [r'$\rm N_{2}$', r'$\rm NH_{3}$', r'$\rm H_{2}$']
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    axs = [ax[i, j] for i in range(2) for j in range(2)]
    for i, item in enumerate(np.array(y_opt).T):
        axs[i].scatter(y=np.array(y_opt).T[i], x=np.array(y_data).T[i], )
        axs[i].plot(np.array(y_data).T[i],
                    np.array(y_data).T[i],
                    color='black')
        axs[i].set_xlabel(xlabel=legend_labels[i] + "-obs mass frac")
        axs[i].set_ylabel(ylabel=legend_labels[i] + "-pred mass frac")
        plt.tight_layout()
    plt.savefig(estimator_name + '/Parity-Plot-MKM')
    for f in ['reactor.yaml', 'thermo_modified.xml', 'thermo_modified.cti']:
        os.remove(f)


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Finished running the parameter estimation in {toc - tic:0.4f} seconds")
