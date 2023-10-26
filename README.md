# petBOA

This is an open-source software package to facilitate parameter estimation using 
Bayesian optimization implemented by the NEXTorch/BOTorch framework.

<img src="petBOA_logo.jpg" alt="petBOA logo" style="height: 250px; width:375px;"/>

Documentation
-------------

At this point detailed documentation doesn't exist for this project. However, we attempt to describe all the folders 
that exist in this project, and also give a short summary of each example present. 

### Folder Tree for the main project

    .
    ├── petboa                  # Source files: classes and methods for BO based parameter estimation
    ├── examples                # All example templates with respective input data and results 
    ├── LICENSE.md
    ├── setup.py
    └── README.md

    
### Examples folder tree 
This describes the folder tree for the templates. 

```bash
.
.
├── example_1_rosenbrock                       # Generic PE template for a parametrized model 
│   ├── evaluate_GSA                           # Global sensitivity analysis using SALib to identify sensitive params
│   ├── petboa                                 # Fit parameters using petBOA
│   └── scipy                                  # Fit parameters using SciPy Nelder-Mead
├── example_2_python_EDH_model                 # Python Macrokinetic model: Batch Reactor for the ethane dehydrogenation 
│   ├── evaluate_gsa                           # Global sensitivity analysis using SALib to identify sensitive params
│   ├── gen_data                               # Generate simulated data used for fitting the macrokinetic model
│   ├── petboa_all_params                      # Fit parameters using petBOA
│   ├── petboa_noisy_data                      # Fit parameters with noisy simulated data using petBOA
│   └── scipy_all_params                       # Fit parameters using SciPy
├── example_3_omkm_NH3_model                   # OpenMKM Black-box Microkinetic model: CSTR reactor NH3 MKM 
│   ├── gen_data                               # Generate simulated data
│   ├── petboa_fit_params                      # Fit using petBOA
│   ├── scipy_fit_params                       # Fit using SciPy
│   └── sensitivity_analysis                   # Local and Global sensitivity analysis identify sensitive params                    
└── example_4_12DCA_model                      # OpenMKM Black-box Microkinetic model: CSTR reactor 1,2 DCA (DOI:  )   
    ├── evaluate_DRC_fullmodel                 # Degree of rate-control analysis identify sensitive params                    
    ├── full-model                             # Fit parameters with the full order DCA MKM using petBOA and SciPy's Differential Evolution
    ├── inputs                                    
    └── reduced-model                          # Fit parameters with the reduced order DCA MKM using petBOA
```
More detailed documentation will be added once the source code and examples are finalized. 

Developers
----------

-  Yifan Wang (wangyf@udel.edu)
-  Sashank Kasiraju (skasiraj@udel.edu)
-  Gerhard Wittreich (wittregr@udel.edu)

Dependencies
------------

-  `Python`
-  `NEXTorch` : Used for implementing the parameter estimation with BO optimization framework
-  `PyTorch` : Used for tensor operations with GPU and autograd support
-  `GPyTorch` : Used for training Gaussian Processes
-  `BoTorch` : Used for providing Bayesian Optimization framework
-  `Matplotlib`: Used for generating plots
-  `PyDOE2`: Used for constructing experimental designs
-  `Numpy`: Used for vector and matrix operations
-  `Scipy`: Used for curve fitting
-  `Pandas`: Used to import data from Excel or CSV files
-  `openpyxl`: Used by Pandas to import Excel files
-  `pytest`: Used for unit tests


Getting Started
---------------

1. It will be a good idea to install all the dependencies in a virtual python enviroment such as [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to make sure you don't corrupt system python.
2. For instance if you are doing this using conda env these steps should give you the required configuration. 
```
conda create --name petboa python==3.9
conda activate petboa
```
3. Install petboa and the dependencies simply using pip.
   `pip install petboa`
4. Install any missing dependencies using pip. Usually installing nextorch should install all other depedencies::

    `pip install nextorch`

5. This step is optional. If pip and git aren't installed in the conda environment:
```
conda install -c anaconda pip
conda install -c anaconda git
```
6. The examples and the source-code can also be downloaded from GitHub. 
```
https://github.com/VlachosGroup/petBOA.git
```
7. Run the examples in the examples folder to look at the four parameter estmation templates that we have created. 

License
-------

This project is licensed under the MIT License - see the LICENSE.md.
file for details.


Contributing
------------

If you have a suggestion or find a bug, please post to our `Issues` page on GitHub. 

Questions
---------

If you are having issues, please post to our `Issues` page on GitHub.

Funding
-------

This material is based upon work supported by the Department of Energy's Office 
of Energy Efficient and Renewable Energy's Advanced Manufacturing Office under 
Award Number DE-EE0007888-9.5.

Acknowledgements
------------------

Max Cohen - For the ethane dehydrogenation model and several uselful discussions. 


Publications
------------
Aritcle submitted. DOI: To-be-added. 
