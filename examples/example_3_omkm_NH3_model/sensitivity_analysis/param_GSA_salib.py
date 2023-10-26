"""
Local Sensitivity Analysis to identify the
sensitive parameters for parameter perturbations
for the NH3 MKM in OpenMKM
"""
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze
from SALib.sample import sobol
from petboa.modelwrappers import ModelWrapper
from petboa.omkm import OMKM
from petboa.utils import RMSE

from make_omkm_inputs import edit_thermo_xml, edit_reactor_yaml, load_thermo_objects, update_param_thermo


def loss_func(self,
              params=None,
              return_parity=False,
              **kwargs):
    """
    Customized loss function specific to this problem
    """
    loss = 0.0
    reg_loss = 0.0
    lamda = kwargs['lamda']
    alpha = kwargs['alpha']
    y_data = []
    y_predict = []
    x_data = []
    _e = []
    _reg_e = []

    # Reload the base thermodynamic model
    os.chdir(self.model.wd_path)
    [reactions, species, phases, units, interactions] = pickle.load(open(kwargs['thermo_data'], 'rb'))
    param_keys = self.para_names

    new_species = update_param_thermo(new_species=species,
                                      param_keys=param_keys,
                                      params=params)
    thermo_data = [reactions, new_species, phases, units, interactions]

    for i in range(self.n_trials):
        _error = 0.0
        _reg_error = 0.0
        p = self.x_inputs[i][0]
        T = self.x_inputs[i][1]
        q = self.x_inputs[i][2]
        y_exp = self.y_groundtruth[i][:]

        os.chdir(self.model.wd_path)
        edit_reactor_yaml(filename="reactor.yaml", p=p, t=T, q=q,
                          y=[])

        edit_thermo_xml(cti_path=self.model.thermo_file.split('.')[0] + '.cti',
                        thermo_data=thermo_data,
                        P=p,
                        T=T,
                        use_motz_wise=True)

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
            y_model = pd.read_csv("gas_mass_ss.csv").iloc[1][['N2', 'NH3', 'H2']].to_numpy()
            y_data.append(y_exp)
            y_predict.append(y_model)
            x_data.append([i, T, p])
            if lamda is None:
                _reg_error = 0.0
            else:
                _reg_error = lamda * np.mean(np.abs(params[:]))
            _error = alpha * RMSE(y_model, y_exp)
            _e.append(_error)
            _reg_e.append(_reg_error)
    # end customization specific to the problem

    loss = np.mean(_e)
    reg_loss = np.mean(_reg_e)
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss, reg_loss])
    self.param_evolution.append([self.call_count] + list(params))
    os.chdir(self.model.wd_path)
    if return_parity:
        return x_data, y_data, y_predict
    else:
        return loss


def main():
    # Define the path to the OpenMKM (omkm) executable
    omkm_path = "/Users/skasiraj/software/openmkm/bin/omkm"
    cwd = os.getcwd()
    omkm_instance = OMKM(exe_path=omkm_path,
                         wd_path=cwd,
                         save_folders=False,
                         )
    data = pd.read_csv(filepath_or_buffer="../gen_data/all_data.csv")
    data.rename(columns={"Unnamed: 0": "exp_no"}, inplace=True)
    data = data.sample(n=10)
    x_input = data[['pressure(atm)', 'temperature(K)', 'vol_flow_rate(cm3/sec)']].to_numpy()
    y_response = data[['N2_massfrac', 'NH3_massfrac', 'H2_massfrac']].to_numpy()

    thermo_data = load_thermo_objects('../gen_data/inputs/NH3_Input_Data.xlsx')
    with open('thermo_data.obj', 'wb') as file:
        pickle.dump(thermo_data, file)
    file.close()

    spec_names = ['N2(T)', 'N(T)', 'H(T)', 'NH3(T)', 'NH2(T)', 'NH(T)',
                  'N2(S)', 'N(S)', 'H(S)', 'NH3(S)', 'NH2(S)', 'NH(S)',
                  'TS4_N2(T)', 'TS4_N2(S)',
                  ]
    parameter_range = [[-5.0, 5.0]] * len(spec_names)
    estimator_name = 'GSA-results'
    if not os.path.isdir(estimator_name):
        os.mkdir(estimator_name)
    os.chdir(estimator_name)
    a = open('output.log', mode='w')
    param_res = []
    full_df = pd.DataFrame()

    ModelWrapper.loss_func = loss_func  # Connect loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=omkm_instance,  # openmkm wrapper with the "run" method
                           para_names=spec_names,
                           name=estimator_name,
                           )
    wrapper.input_data(x_inputs=x_input,
                       n_trials=len(data),
                       y_groundtruth=y_response)
    problem = {
        'num_vars': len(spec_names),
        'names': spec_names,
        'bounds': parameter_range
    }

    param_values = sobol.sample(problem, 4)
    print("Total number of sensitivity run samples {}".format(param_values.shape[0]))
    a.write("Total number of sensitivity run samples {} \n".format(param_values.shape[0]))
    Y = np.zeros([param_values.shape[0]])
    for i, X in enumerate(param_values):
        Y[i] = wrapper.loss_func(params=X,
                                 alpha=1.0,
                                 lamda=1.0,
                                 thermo_data='thermo_data.obj',
                                 return_parity=False
                                 )
        print("Finished evaluating {} sample with loss {}".format(i, Y[i]))
        a.write("Finished evaluating {} sample with loss {} \n".format(i, Y[i]))

    os.chdir(estimator_name)
    history = np.append(param_values, np.reshape(Y, (len(Y), 1)), axis=1)
    hist_df = pd.DataFrame(data=history,
                           columns=spec_names + ["loss"])
    hist_df.to_csv("sobol_sample.csv")
    si = analyze(problem, Y)
    a.write("\n \nFirst Order Sensitivity Indices are:\n {} \n".format(si['S1']))
    a.write("Total Order Sensitivity Indices are:\n {} \n".format(si['ST']))

    data = {"ST": pd.Series(data=si['ST'],
                            index=spec_names),
            "S1": pd.Series(data=si['S1'],
                            index=spec_names)}
    df = pd.DataFrame(data=data)
    print(df)
    a.write("\nGSA Coefficients \n \n")
    a.write(df.to_string())
    df.to_csv('results.csv')
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
