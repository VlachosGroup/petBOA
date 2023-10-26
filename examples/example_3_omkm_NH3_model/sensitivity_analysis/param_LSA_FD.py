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

    estimator_name = 'LSA-results'
    if not os.path.isdir(estimator_name):
        os.mkdir(estimator_name)
    os.chdir(estimator_name)
    a = open('output.log', mode='w')
    spec_names = ['N2(T)', 'N(T)', 'H(T)', 'NH3(T)', 'NH2(T)', 'NH(T)',
                  'N2(S)', 'N(S)', 'H(S)', 'NH3(S)', 'NH2(S)', 'NH(S)',
                  'TS4_N2(T)', 'TS4_N2(S)',
                  ]
    print("Total number of params {} are checked for sensitivity {}".format(len(spec_names),
                                                                            spec_names))
    a.write("Total number of params {} are checked for sensitivity {} \n".format(len(spec_names),
                                                                                 spec_names))
    ModelWrapper.loss_func = loss_func  # Connect loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=omkm_instance,  # openmkm wrapper with the "run" method
                           para_names=spec_names,
                           name=estimator_name,
                           )
    wrapper.input_data(x_inputs=x_input,
                       n_trials=len(data),
                       y_groundtruth=y_response)

    h = 10.0  # in percentage change for parameter
    x_param = 1.0  # sensitivity estimated at a perturbation of 1kJ/mol for each param
    dlnf_dlnparam = np.zeros(len(spec_names))

    for i, spec in enumerate(spec_names):
        print("LSA {} for species {}".format(i, spec))
        a.write("LSA {} for species {} \n".format(i, spec))
        params = np.zeros(len(spec_names)) + x_param
        params[i] *= (100.0 + h) / 100.0
        f_xplush = wrapper.loss_func(params=params,
                                     alpha=1.0,
                                     lamda=1.0,
                                     thermo_data='thermo_data.obj',
                                     return_parity=False)

        params = np.zeros(len(spec_names)) + x_param
        params[i] *= (100.0 - h) / 100.0
        f_xminush = wrapper.loss_func(params=params,
                                      alpha=1.0,
                                      lamda=1.0,
                                      thermo_data='thermo_data.obj',
                                      return_parity=False)
        dolnf = np.log(np.abs(f_xplush)) - np.log(np.abs(f_xminush))
        dolnp = np.log(x_param * (100.0 + h) / 100.0) - np.log(x_param * (100.0 - h) / 100.0)
        if np.isinf(dolnf):
            dlnf_dlnparam[i] = 0.0
        else:
            dlnf_dlnparam[i] = dolnf / dolnp
        a.write("f(x+h) {} f(x-h) {} dln-f by dln-param[i] {} \n".format(f_xplush,
                                                                         f_xminush,
                                                                         dlnf_dlnparam[i]))
    data = {"1st order Grad": pd.Series(data=dlnf_dlnparam,
                                        index=spec_names),
            }
    df = pd.DataFrame(data=data)
    print(df)
    a.write("\nNSC/LSA Coefficients \n \n")
    a.write(df.to_string())
    os.chdir(cwd)
    if not os.path.exists(estimator_name):
        os.mkdir(estimator_name)
    os.chdir(estimator_name)
    df.to_csv('lsa_results.csv')
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
