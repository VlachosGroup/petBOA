Documentation
-------------

We describe the sub-folders in the *Example 1* which demonstrates the use of petBOA for a Rosenbrock test problem. 

### Folder Tree for Example 1

```
.
├── evaluate_GSA
│   └── rosenbrock_GSA.py
├── petboa
│   ├── params.xlsx
│   └── rosen_petBOA.py
└── scipy
    ├── history.csv
    ├── parameter_fits.csv
    ├── params.xlsx
    └── rosen_min_scipy.py
```

1. To run the global sensitivity analysis, navigate ot the `evaluate_GSA` folder and run the python script `rosenbrock_GSA.py`
2. To estimate parameters *a* and *b* using the petBOA's optimizer navigate to the `petBOA` folder and run the python script `rosen_petBOA.py`
3. To estimate parameters *a* and *b* using the SciPy's optimizer navigate to the `scipy` folder and run the python script `rosen_min_scipy.py`
