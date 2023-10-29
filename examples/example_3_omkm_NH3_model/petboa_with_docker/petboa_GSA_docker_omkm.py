"""
Fit 4 species identified from GSA
for parameter perturbations using the
petBOA optimizer for the NH3 MKM in OpenMKM. 
Run OpenMKM using Docker. 
"""
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from docker.errors import DockerException

from petboa.modelwrappers import ModelWrapper
from petboa.omkm import OMKM
from petboa.plots import plot_parity
from petboa.utils import RMSE, parse_param_file
from petboa.optimizer import BOOptimizer

from make_omkm_inputs import edit_thermo_xml, edit_reactor_yaml, load_thermo_objects, update_param_thermo
import docker


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

        try:
            if self.model.process_instance is not None:
                code = self.model.process_instance.returncode
            else:
                code = self.model.docker_exec_code
        except:
            print("OpenMKM or Docker not configured correctly")

        if not code == 0:
            print("Model {} Failed \n {}".format(i, self.model.stderr))
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
    print("In Optimization iteration {} Loss is {:.5f}".format(self.call_count, loss))
    os.chdir(self.model.wd_path)
    if return_parity:
        return x_data, y_data, y_predict
    else:
        return loss


def main():
    cwd = os.getcwd()
    # Start a Docker client
    try:
        client = docker.from_env()
    except DockerException:
        raise RuntimeError("Either the Docker daemon isn't running or the Docker "
                           "container is not properly setup")

    # Define the Docker container's configuration
    container_config = {
        'image': 'vlachosgroup/openmkm',
        'detach': True,
        'tty': True,
        'working_dir': '/data',
        'volumes': {
            os.path.abspath('.'): {
                'bind': '/data',
                'mode': 'rw',
            }
        },
    }
    # Start the Docker container with the specified command
    container = client.containers.run(**container_config)

    # Create an OpenMKM Docker instance
    omkm_instance = OMKM(wd_path=cwd,
                         save_folders=False,
                         docker=container,
                         verbose=True,
                         )

    data = pd.read_csv(filepath_or_buffer="../gen_data/all_data.csv")
    data.rename(columns={"Unnamed: 0": "exp_no"}, inplace=True)
    data = data[:]
    x_input = data[['pressure(atm)', 'temperature(K)', 'vol_flow_rate(cm3/sec)']].to_numpy()
    y_response = data[['N2_massfrac', 'NH3_massfrac', 'H2_massfrac']].to_numpy()

    thermo_data = load_thermo_objects('../gen_data/inputs/NH3_Input_Data.xlsx')
    with open('thermo_data.obj', 'wb') as file:
        pickle.dump(thermo_data, file)
    file.close()

    para_ground_truth = {}

    # species_to_change = ['N2(T)', 'N(T)', 'H(T)', 'NH3(T)', 'NH2(T)', 'NH(T)',
    #                      'N2(S)', 'N(S)', 'H(S)', 'NH3(S)', 'NH2(S)', 'NH(S)',
    #                      'TS4_N2(T)', 'TS4_N2(S)',
    #                      ]
    # Select only the sensitive species from GSA
    # species_to_change = ['N(S)', 'H(S)', 'NH2(S)', 'TS4_N2(S)']

    species_to_change, parameter_range = parse_param_file("params_GSA.xlsx")

    # You can remove a parameter from fitting by not passing the bounds
    # and just passing a nominal value. Below parameters 1, 2, 3 and not
    # tuned and fixed at 6.959, -4.629, -16.88
    # species_to_change = ['N2(T)', 'N(T)', 'TS4_N2(T)', 'N2(S)', 'N(S)', 'TS4_N2(S)']
    # parameter_range = [6.959, -4.629, -16.88, [-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]]

    estimator_name = 'omkm_petboa_GSA'
    if not os.path.isdir(estimator_name):
        os.mkdir(estimator_name)
    os.chdir(estimator_name)
    a = open('output.log', mode='w')
    param_res = []
    full_df = pd.DataFrame()

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
                               para_names=species_to_change,
                               name=estimator_name,
                               )
        wrapper.input_data(x_inputs=x_input,
                           n_trials=len(data),
                           y_groundtruth=y_response)
        optimizer = BOOptimizer(estimator_name)
        n_iter = 1
        alpha = 10.0
        lamda = 0.001
        X_opt, loss_opt, Exp = optimizer.optimize(objective_func=wrapper.loss_func,
                                                  para_ranges=parameter_range,
                                                  n_sample_multiplier=1,
                                                  n_iter=n_iter,
                                                  log_flag=True,
                                                  lamda=lamda,
                                                  alpha=alpha,
                                                  thermo_data='thermo_data.obj',
                                                  )
        a.write("Objective function called {} times \t \n".format(wrapper.call_count))
        for i, name in enumerate(para_ground_truth.keys()):
            a.write("For Parameters {} are Optimum value is {} \n".format(name, X_opt[i]))
        end_time = time.time()
        a.write("Total Time: {} sec \n".format(end_time - start_time))
        print("Objective function called {} times".format(wrapper.call_count))
        print("Parameters are {} \n".format(X_opt))
        print("Total time in sec {} \n".format(end_time - start_time))

        df1 = pd.DataFrame(data=wrapper.loss_evolution,
                           columns=['Run No', 'Loss', 'Reg. Loss'],
                           )
        df1['Repeat'] = repeat
        # print(wrapper.param_evolution, ['Run No'] + species_to_change)
        df2 = pd.DataFrame(data=wrapper.param_evolution,
                           columns=['Run No'] + species_to_change,
                           )
        df1 = df1.merge(df2, how='inner', on='Run No')
        full_df = pd.concat([full_df, df1])
        param_res.append(X_opt)

    os.chdir(estimator_name)
    full_df.to_csv("param_loss_history.csv")
    df3 = pd.DataFrame(data=param_res,
                       columns=species_to_change,
                       )
    df3 = df3.T
    df3['mean'] = df3.mean(axis=1)
    df3['std'] = df3.std(axis=1)
    df3.to_csv("parameter_fits.csv")

    # Visualize the fits and improvement
    # Get initial parity1
    os.chdir(cwd)
    legend_labels = [r'$\rm N_{2}$', r'$\rm NH_{3}$', r'$\rm H_{2}$']
    ModelWrapper.loss_func = loss_func  # Connect loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=omkm_instance,  # openmkm wrapper with the "run" method
                           para_names=species_to_change,
                           name=estimator_name,
                           )
    wrapper.input_data(x_inputs=x_input,
                       n_trials=len(data),
                       y_groundtruth=y_response)

    X_data, Y_data, Y_opt = wrapper.loss_func(params=np.zeros(len(species_to_change)),
                                              alpha=1.0,
                                              lamda=1.0,
                                              thermo_data='thermo_data.obj',
                                              return_parity=True, )
    plot_parity(X_data=X_data, Y_data=Y_data, Y_opt=Y_opt,
                legend_labels=legend_labels,
                estimator_name=estimator_name,
                plot_name='Parity-Plot-Initial-MKM'
                )

    X_data, Y_data, Y_opt = wrapper.loss_func(params=df3['mean'],
                                              alpha=1.0,
                                              lamda=1.0,
                                              thermo_data='thermo_data.obj',
                                              return_parity=True, )

    plot_parity(X_data=X_data, Y_data=Y_data, Y_opt=Y_opt,
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

    # Cleanup and close the Docker client
    container.stop()
    container.remove()


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Finished running the parameter estimation in {toc - tic:0.4f} seconds")
