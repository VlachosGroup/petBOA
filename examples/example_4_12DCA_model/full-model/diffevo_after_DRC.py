"""
Parameter Estimation for the full order 1,2 DCA MKM in OpenMKM
with SciPy's Differential Evolution optimizer by selecting sensitive
parameters using degree of rate control (DRC)
"""
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from petboa.modelwrappers import ModelWrapper
from petboa.omkm import OMKM
from scipy.optimize import differential_evolution

from make_omkm_inputs import edit_thermo_xml, edit_reactor_yaml, load_thermo_objects, update_param_thermo


def loss_func(self,
              params=None,
              *args):
    """
    Customized loss function specific to this problem
    """
    try:
        options = args[0][0]
    except KeyError:
        options = args[0]
    lamda = options['lamda']
    y_data = []
    y_predict = []
    x_data = []
    _e = []
    _reg_e = []

    os.chdir(self.model.wd_path)
    param_keys = self.para_names

    [new_reactions, new_species, phases, units, interactions] = pickle.load(open(options['thermo-pickle'], 'rb'))
    update_param_thermo(reactions=new_reactions,
                        new_species=new_species,
                        param_keys=param_keys,
                        params=params
                        )
    thermo_data = [new_reactions, new_species, phases, units, interactions]

    for i in range(self.n_trials):
        _error = 0.0
        _reg_error = 0.0
        T = self.x_inputs[i][0]
        y = self.x_inputs[i][1:]
        P = 1.5  # atm
        tof_exp = self.y_groundtruth[i][:][0]
        os.chdir(self.model.wd_path)

        edit_reactor_yaml(filename="reactor.yaml", y=y, T=T)

        edit_thermo_xml(cti_path=self.model.thermo_file.split('.')[0] + '.cti',
                        thermo_data=thermo_data,
                        P=P,
                        T=T,
                        use_motz_wise=False,
                        )

        self.model.run(i)
        if not self.model.process_instance.returncode == 0:
            print("Model {} Failed \n {}".format(i, self.model.process_instance.stderr))
            _error += 1.0E10
            _e.append(_error)
            print("Error {} Reg. Error {}".format(_error, _reg_error))
            print("This run failed badly {}".format(self.model.run_number))
            print(params)
            shutil.copy('thermo.xml', '../thermo_' + str(self.model.run_number) + '.xml')
        else:
            if not options['tof-to-fit'] is None:
                tof_model = pd.read_csv("gas_sdot_ss.csv").iloc[1][options['tof-to-fit']].to_numpy()[0]
            else:
                raise Exception("Define species for which TOF has to be fit")

            vol_flowrate = 2. # cm3/s
            reac_vol = 2. # cm3
            abyv = 463.
            site_density = 1.66e-9

            #  Assuming sdot_ss is in the units of kmol/s
            #  TOF = rate/(abyv*volume*site-density)
            tof_model = np.abs(tof_model) * 1000.0  # kmol/s to mol/s
            tof_model = tof_model / (reac_vol * abyv * site_density)  # (mol/s)/((mol/cm2) * (cm2/cm3) * cm3)  == 1/s

            # New average TOF metric
            new_R = 0.082057366080960  # units of R = 	L⋅atm⋅K−1⋅mol−1
            total_moles_in = 1.5 * (reac_vol / 1000) / (new_R * T)  # n = (atm *L) /(L⋅atm⋅K−1⋅mol−1 * K)
            DCA_in, H2_in, Ar_in = y[0], y[1], y[2]
            DCA_MW, H2_MW, Ar_MW, C2H4_MW = 98.959, 2.01588, 39.948, 28.0532
            total_mass_in = total_moles_in * (DCA_in * DCA_MW + H2_in * H2_MW + Ar_in * Ar_MW)
            C2H4_moles_in = 0.0
            C2H4_mass_frac_outlet = pd.read_csv("gas_mass_ss.csv").iloc[-1][options['tof-to-fit']].to_numpy()[0]
            C2H2_moles_out = (C2H4_mass_frac_outlet * total_mass_in) / C2H4_MW
            r_time = reac_vol / vol_flowrate
            avg_rate = (C2H2_moles_out - C2H4_moles_in) / r_time  # in mol/s
            tof_avg = avg_rate / (reac_vol * abyv * site_density)

            print("TOF_Exp {} and TOF_Model {} and TOF_avg {} ".format(tof_exp, tof_model, tof_avg))
            y_data.append(tof_exp)
            y_predict.append(tof_avg)
            x_data.append([i, T, y])
            if lamda is None:
                _error = np.mean(((tof_exp / tof_model) - 1.0) ** 2) * 100
            else:
                _reg_error = lamda * np.mean(np.abs(params[3:]))
                _error += _reg_error + np.mean(((tof_model / tof_exp) - 1.0) ** 2) * 100
                print("Error {} Reg. Error {}".format(_error, _reg_error))
            _e.append(_error)
            _reg_e.append(_reg_error)
    #  end customization specific to the problem
    loss = np.mean(_e)
    reg_loss = np.mean(_reg_e)
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss, reg_loss])
    self.param_evolution.append([self.call_count] + list(params))
    print("Optimization Iteration {} loss {}".format(self.call_count, loss))
    os.chdir(self.model.wd_path)
    if options['return_parity']:
        return x_data, y_data, y_predict
    else:
        return loss


def plot_parity(X_data,
                Y_data,
                Y_opt,
                legend_labels,
                estimator_name,
                plot_name='Parity-Plot-MKM'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(y=np.array(Y_opt), x=np.array(Y_data), )
    ax.plot(np.array(Y_data),
            np.array(Y_data),
            color='black')
    ax.set_xlabel(xlabel=legend_labels[0] + "-Exp")
    ax.set_ylabel(ylabel=legend_labels[0] + "-Model")
    ax.set_xlim([0.0, 1.10 * max(Y_data)])
    ax.set_ylim([0.0, 1.10 * max(Y_data)])
    plt.tight_layout()
    plt.savefig(estimator_name + '/' + plot_name)


def main():
    # Define the path to the OpenMKM (omkm) executable
    omkm_path = "/Users/skasiraj/software/openmkm/bin/omkm"
    cwd = os.getcwd()
    omkm_instance = OMKM(exe_path=omkm_path,
                         wd_path=cwd,
                         save_folders=False,
                         )
    data = pd.read_csv(filepath_or_buffer="../inputs/all_data.csv")
    # data = data.sample(10)
    x_input = data[['Temperature(K)', "y(1,2DCA)", 'y(H2)', 'y(Ar)']].to_numpy()
    y_response = data[['TOF(s-1)']].to_numpy()
    thermo_data = load_thermo_objects('../inputs/12DCA_Input_Data.xlsx')
    with open('thermo_data.obj', 'wb') as file:
        pickle.dump(thermo_data, file)
    file.close()

    tof_to_fit = ['CH2CH2']
    legend_labels = [r'$TOF-C_{2}H_{4}$ (1/s)']

    max_deviation = 40.0  # kJ/mol
    para_ground_truth = {'TS12': 0.0,
                         'TS16': 0.0,
                         'TS47': 0.0,
                         'TS49': 0.0,
                         'Cl*': 0.0,
                         'H*': 0.0,
                         'CH3CH3*': 0.0,
                         'CH3CH2*': 0.0,
                         'ClCH2CH2Cl*': 0.0,
                         'ClCH2CH2*': 0.0,
                         'CH2CH2*': 0.0,
                         }
    parameter_range = [[-1.0 * max_deviation + para_ground_truth[_],
                        max_deviation + para_ground_truth[_]]
                       for _ in para_ground_truth.keys()]

    estimator_name = 'diffevo_after_DRC'
    if not os.path.isdir(estimator_name):
        os.mkdir(estimator_name)

    os.chdir(estimator_name)
    a = open('output.log', mode='w')

    full_df = pd.DataFrame()
    param_res = []

    for repeat in range(1):
        start_time = time.time()
        print("###################### \n")
        print("Repeat {} \n".format(str(repeat)))
        print("###################### \n")
        a.write("###################### \n")
        a.write("Repeat {} \n".format(str(repeat)))
        a.write("###################### \n")

        ModelWrapper.loss_func = loss_func  # Connect loss function handle to the Model Wrapper Class
        wrapper = ModelWrapper(model_function=omkm_instance,  # openmkm wrapper with the "run" method
                               para_names=list(para_ground_truth.keys()),
                               name=estimator_name,
                               )
        wrapper.input_data(x_inputs=x_input,
                           n_trials=len(data),
                           y_groundtruth=y_response)

        loss_options = ({'lamda': 0.0,
                         'thermo-pickle': 'thermo_data.obj',
                         'tof-to-fit': tof_to_fit,
                         'return_parity': False
                         })
        res = differential_evolution(func=wrapper.loss_func,
                                     bounds=parameter_range,
                                     args=([loss_options]),
                                     strategy='best1bin',
                                     maxiter=1,
                                     popsize=3,
                                     )
        end_time = time.time()
        a.write("Objective function called {} {} times \n".format(wrapper.call_count, res.nfev))
        for i, name in enumerate(para_ground_truth.keys()):
            a.write("For Parameters {} are Optimum value is {} \n".format(name, res['x'][i]))
        a.write("Parameters are {} \n".format(res['x']))
        a.write("Total time taken in sec {} \n \n \n".format(end_time - start_time))
        a.write("Optimizer results {}".format(res))

        df1 = pd.DataFrame(data=wrapper.loss_evolution,
                           columns=['Run No', 'Loss', 'Reg. Loss'],
                           )
        df1['Repeat'] = repeat
        df2 = pd.DataFrame(data=wrapper.param_evolution,
                           columns=['Run No'] + list(para_ground_truth.keys()),
                           )
        df1 = df1.merge(df2, how='inner', on='Run No')
        full_df = pd.concat([full_df, df1])
        param_res.append(res['x'])

    os.chdir(estimator_name)
    full_df.to_csv("param_loss_history.csv")
    df3 = pd.DataFrame(data=param_res,
                       columns=list(para_ground_truth.keys()),
                       )
    df3 = df3.T
    df3['mean'] = df3.mean(axis=1)
    df3['std'] = df3.std(axis=1)
    df3.to_csv("parameter_fits.csv")

    loss_options = ({'lamda': 0.0,
                     'thermo-pickle': 'thermo_data.obj',
                     'tof-to-fit': tof_to_fit,
                     'return_parity': True
                     })
    X_data, Y_data, Y_opt = wrapper.loss_func(res['x'],
                                              [loss_options],)

    plot_parity(X_data=X_data,
                Y_data=Y_data,
                Y_opt=Y_opt,
                legend_labels=legend_labels,
                estimator_name=estimator_name,
                )

    # Delete intermediate files
    os.chdir(cwd)
    for f in ["thermo.cti", "thermo_data.obj", "thermo.xml", "reactor.yaml", "run"]:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Finished running the parameter estimation in {toc - tic:0.4f} seconds")
