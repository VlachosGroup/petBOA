"""
Local Sensitivity Analysis to identify the
sensitive parameters for parameter perturbations
for the 1,2 DCA MKM in OpenMKM
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
    x_data = []
    _rate = []
    # Reload the base thermodynamic model
    os.chdir(self.model.wd_path)
    [reactions, species, phases, units, interactions] = pickle.load(open(kwargs['thermo_data'], 'rb'))
    param_keys = self.para_names

    for i in range(self.n_trials):
        T = self.x_inputs[i][0]
        y = self.x_inputs[i][1:]
        Q = params[0]
        P = 1.5  # atm

        os.chdir(self.model.wd_path)
        edit_reactor_yaml(filename="reactor.yaml", y=y, T=T)

        new_reactions = update_param_thermo(reactions=reactions,
                                            units=units,
                                            T=T,
                                            P=P,
                                            perturb=kwargs['perturb']
                                            )
        thermo_data = [new_reactions, species, phases, units, interactions]

        edit_thermo_xml(cti_path=self.model.thermo_file.split('.')[0] + '.cti',
                        thermo_data=thermo_data,
                        P=P,
                        T=T,
                        use_motz_wise=False,
                        )

        self.model.run(i)
        if not self.model.process_instance.returncode == 0:
            print("Model {} Failed \n {}".format(i, self.model.process_instance.stderr))
            print("This run failed badly {}".format(self.model.run_number))
            shutil.copy('thermo.xml', '../thermo_' + str(self.model.run_number) + '.xml')
        else:
            if not kwargs['tof_to_fit'] is None:
                tof_model = pd.read_csv("gas_sdot_ss.csv").iloc[1][kwargs['tof_to_fit']].to_numpy()[0]
            else:
                raise Exception("Define species for which TOF has to be fit")
            _rate.append(np.abs(tof_model))
            x_data.append([i, T, Q, y])
    #  end customization specific to the problem
    self.call_count += 1
    os.chdir(self.model.wd_path)
    return _rate


def main():
    # Define the path to the OpenMKM (omkm) executable
    omkm_path = "/Users/skasiraj/software/openmkm/bin/omkm"
    cwd = os.getcwd()
    omkm_instance = OMKM(exe_path=omkm_path,
                         wd_path=cwd,
                         save_folders=False,
                         )
    data = pd.read_csv(filepath_or_buffer="../inputs/all_data.csv")
    data = data.sample(5)
    x_input = data[['Temperature(K)', "y(1,2DCA)", 'y(H2)', 'y(Ar)']].to_numpy()
    y_response = data[['TOF(s-1)']].to_numpy()
    thermo_data = load_thermo_objects('../inputs/12DCA_Input_Data.xlsx')
    [reactions, new_species, phases, units, interactions] = thermo_data
    with open('thermo_data.obj', 'wb') as file:
        pickle.dump(thermo_data, file)
    file.close()

    tof_to_fit = ['CH2CH2']
    para_ground_truth = {'Q': 10.00}  # cm3/s

    estimator_name = 'DRC-results'
    if not os.path.isdir(estimator_name):
        os.mkdir(estimator_name)
    os.chdir(estimator_name)
    a = open('output.log', mode='w')

    ModelWrapper.loss_func = loss_func  # Connect loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=omkm_instance,  # openmkm wrapper with the "run" method
                           para_names=list(para_ground_truth.keys()),
                           name=estimator_name,
                           )
    wrapper.input_data(x_inputs=x_input,
                       n_trials=len(data),
                       y_groundtruth=y_response)
    drc_f = []
    drc_c = []
    h = 0.0001
    f_x = []
    for i, reaction in enumerate(reactions[:]):
        params = np.array(list(para_ground_truth.values()))
        perturb_none = 1.00

        f_x.append(wrapper.loss_func(params=params,
                                     thermo_data='thermo_data.obj',
                                     lamda=None,
                                     tof_to_fit=tof_to_fit,
                                     perturb=(i, perturb_none)
                                     ))
        params = np.array(list(para_ground_truth.values()))
        perturb_plus = 1.00 + h
        f_xplush = wrapper.loss_func(params=params,
                                     thermo_data='thermo_data.obj',
                                     lamda=None,
                                     tof_to_fit=tof_to_fit,
                                     perturb=(i, perturb_plus)
                                     )
        perturb_minus = 1.00 - h
        f_xminush = wrapper.loss_func(params=params,
                                      thermo_data='thermo_data.obj',
                                      lamda=None,
                                      tof_to_fit=tof_to_fit,
                                      perturb=(i, perturb_minus)
                                      )
        dolnf_p = (np.log(f_xplush) - np.log(f_x[i]))
        dolnp_p = (np.log(perturb_plus) - np.log(1.0))
        dolnf_m = (np.log(f_x) - np.log(f_xminush))
        dolnp_m = (np.log(1.0) - np.log(perturb_minus))
        dolnf_c = (dolnf_p / dolnp_p + dolnf_m / dolnp_m) / 2.0
        drc_c.append(dolnf_c[0])

        dolnf_p = (np.log(f_xplush) - np.log(f_x))
        dolnp_p = (np.log(perturb_plus) - np.log(1.0))
        dolnf_f = (dolnf_p / dolnp_p)
        drc_f.append(dolnf_f[0])
        print("DRC from reaction {} {}".format(i, reaction))
        print("For Exp:1 reaction {} f(x+h) {} f(x) {} f(x-h) {} doln(f)_fd {} doln(f)_cd {}".
              format(reaction.to_string(), f_xplush[0], f_x[0][0], f_xminush[0], dolnf_f[0][0], dolnf_c[0][0]))
        a.write("For Exp:1 reaction {} f(x+h) {} f(x) {} f(x-h) {} doln(f) {} DRC(i) {} \n".
                format(reaction.to_string(), f_xplush[0], f_x[0][0], f_xminush[0], dolnf_f[0][0], dolnf_c[0][0]))

    os.chdir(estimator_name)
    df = pd.DataFrame(drc_c)
    df.index = [r.to_string() for r in reactions[:]]
    df.columns = ['DRC-Expt-' + str(i) for i in range(wrapper.n_trials)]
    df['mean_DRC'] = df.mean(axis=1)
    df['min_DRC'] = df.min(axis=1)
    df['max_DRC'] = df.max(axis=1)
    a.write("Central Difference based \n")
    a.write(df.to_string())
    df.to_csv('DRC_central_diff_' + str(h) + '.csv')

    df = pd.DataFrame(drc_f)
    df.index = [r.to_string() for r in reactions[:]]
    df.columns = ['DRC-Expt-' + str(i) for i in range(wrapper.n_trials)]
    df['mean_DRC'] = df.mean(axis=1)
    df['min_DRC'] = df.min(axis=1)
    df['max_DRC'] = df.max(axis=1)
    a.write("\n \nForward Difference  \n")
    a.write(df.to_string())
    df.to_csv('DRC_forward_diff_' + str(h) + '.csv')

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
