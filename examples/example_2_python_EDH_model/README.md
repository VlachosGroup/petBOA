Documentation
-------------

We describe the sub-folders in the *Example 2* which demonstrates the use of petBOA for a Python macrokinetic model. 

### Folder Tree for Example 1

```
.
├── evaluate_gsa
│   ├── edh_batch_GSA.py
│   ├── ethane_model.py
├── gen_data
│   ├── ethane_model.py
│   ├── expt_data.csv
│   ├── noisy_expt_data.csv
│   ├── simulated_data.py
│   └── simulated_noisy_data.py
├── petboa_all_params
│   ├── ethane_model.py
│   ├── params.xlsx
│   ├── petboa_all_params.py
├── petboa_noisy_data
│   ├── ethane_model.py
│   ├── params.xlsx
│   └── petboa_fit_noisy_data.py
└── scipy_all_params
    ├── ethane_model.py
    ├── params.xlsx
    ├── scipy-fit-all-params
    └── scipy_all_params.py
```

Note: In all the sub-folders the Python script `ethane_model.py` contains the Python model which is imported used by all other scripts. 
Note: In all the sub-folders the 'expt_data.csv contains the simulated data which is used as ground-truth for the parameter estimation purposes. 
Note: In all the sub-folders the Excel file `params.xlsx` contains the model parameter names and the bounds used by the parameter estimation scripts. 

1. To run the global sensitivity analysis, navigate ot the `evauate_GSA` folder and run the python script `edh_batch_GSA.py`. 
2. The `gen_data` folder contains the scripts and input files required to generate simulated data used for parameter estimation. Note: To run the example parameter estimation scripts existing data can be used i.e., no scripts have to be run.
3. To estimate parameters using petBOA run the `petboa_all_params.py` in the folder `petboa_all_params`. Similarly, the example can be repeated for noisy data as shown in the folder `scipy_all_params`. 
4. To estimate parameters using the SciPy's optimizer navigate to the `scipy_all_params` folder and run the python script `scipy_all_params.py`

