Documentation
-------------

We describe the sub-folders in the *Example 3* which demonstrates the use of petBOA for an OpenMKM microkinetic model based on the article DOI: 

### Folder Tree for Example 4

```
.
├── evaluate_DRC_fullmodel
│   ├── DRC_full_model.py
│   └── make_omkm_inputs.py
├── full-model
│   ├── diffevo_after_DRC
│   ├── diffevo_after_DRC.py
│   ├── make_omkm_inputs.py
│   ├── petboa_afterDRC.py
├── inputs
│   ├── 12DCA_Input_Data.xlsx
│   ├── 12DCA_ThermoData.xlsx
│   └── all_data.csv
└── reduced-model
    ├── make_omkm_inputs.py
    ├── petboa_reduced_model
    ├── petboa_reduced_model.py
```

Note: In all the sub-folders the Python script `make_omkm_inputs.py` contains the Python model which is used to make OpenMKM input files. Users don't need to run it, it is automatically imported and used by all other scripts. 
Note: The inputs folders contains the 'all_data.csv file and other input files required to run OpenMKM. Scripts automatically use it using relative paths, as long as they are run from thier original location.  

1. To run the degree of rate control (DRC) analysis for the full-order MKM navigate ot the `evaluate_DRC_fullmodel` folder and run the python script `DRC_full_model.py`. 
2. To estimate parameters using petBOA for the full-order model after getting insights from global-sensitivity (DRC) (see the journal article example 4 for more info) run the `petboa_afterDRC.py` in the folder `full-model`.
3. Similarly, the example can be repeated for SciPy's differential evolution using `diffevo_after_DRC.py`
4. For the reduced-order model run the script `petboa_reduced_model.py` instead in the folder `reduced-model`.
