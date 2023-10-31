Documentation
-------------

We describe the sub-folders in the *Example 3* which demonstrates the use of petBOA for an OpenMKM microkinetic model. 

### Folder Tree for Example 3

```
.
├── gen_data
│   ├── all_data.csv
│   ├── param_devs.csv
│   └── peturb_thermo_data_gen.py
├── petboa_fit_params
│   ├── make_omkm_inputs.py
│   ├── params.xlsx
│   ├── params_GSA.xlsx
│   ├── petboa_fit_params_GSA.py
│   └── petboa_fit_params_PEI.py
├── petboa_with_docker
│   ├── make_omkm_inputs.py
│   ├── params_GSA.xlsx
│   └── petboa_GSA_docker_omkm.py
├── scipy_fit_params
│   ├── make_omkm_inputs.py
│   ├── params.xlsx
│   └── scipy_fit_omkm_model.py
└── sensitivity_analysis
    ├── make_omkm_inputs.py
    ├── param_GSA_salib.py
    └── param_LSA_FD.py
```

Note: In all the sub-folders the Python script `make_omkm_inputs.py` contains the Python model which is used to make OpenMKM input files. Users don't need to run it, it is automatically imported and used by all other scripts. 

Note: In all the sub-folders the `all_data.csv` contains the simulated data which is used as ground-truth for the parameter estimation purposes. 

Note: In all the sub-folders the Excel file `params**.xlsx` contains the model parameter names and the bounds used by the parameter estimation scripts. 

Note: Running examples with OpenMKM models requires OpenMKM installed on the computer. The Python scripts have to be modified to include the path to OpenMKM. 

1. The `gen_data` folder contains the scripts and input files required to generate simulated data used for parameter estimation. Note: To run the example parameter estimation scripts existing data can be used i.e., no scripts have to be run.
2. To estimate parameters using petBOA after getting insights from global-sensitivity (GSA) (see the journal article example 3 for more info) run the `petboa_fit_params_GSA.py` in the folder `petboa_fit_params`. Similarly, the example can be repeated for the partial equilibrium index (PEI) based parameters using `petboa_fit_params_PEI.py`
3. To estimate parameters using the SciPy's optimizer navigate to the `scipy_fit_params` folder and run the python script `scipy_fit_omkm_model.py`
4. To use Docker to run OpenMKM with petBOA see example `petboa_GSA_docker_omkm.py` in the folder `petboa_with_docker`.
Note: Running this example requires Docker service to be installed on the machine and Docker daemon running in the background. 
5. To run the local/global sensitivity analysis, navigate ot the `sensitivity_analysis` folder and run the python scripts `   param_GSA_salib.py or param_LSA_FD.py`. 
