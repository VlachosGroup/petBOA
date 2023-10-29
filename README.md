# petBOA

petBOA is an open-source Python-based Parameter Estimation Tool utilizing Bayesian Optimization with a unique wrapper interface for gradient-free parameter estimation of expensive black-box kinetic models. We provide examples for Python macrokinetic and microkinetic modeling (MKM) tools, such as OpenMKM. petBOA leverages surrogate Gaussian processes to approximate and minimize the objective function designed for parameter estimation. Bayesian Optimization (BO) is implemented using the open-source NEXTorch/BoTorch toolkit. petBOA employs local and global sensitivity analyses to identify important parameters optimized against experimental data, and leverages pMuTT for consistent kinetic and thermodynamic parameters while perturbing species binding energies within the typical error of conventional DFT exchange-correlation functionals (20-30 kJ/mol).

<img src="petBOA_logo.jpg" alt="petBOA logo" style="height: 250px; width:375px;"/>

Documentation
-------------

We describe all the folders that exist in this repository and also give a short summary of each example present. More detailed documentation will be added once the source code and examples are finalized. 

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
│   ├── sensitivity_analysis                   # Local and Global sensitivity analysis identify sensitive params  
│   └── petboa_with_docker                     # Example to run OpenMKM using Docker SDK for Python
└── example_4_12DCA_model                      # OpenMKM Black-box Microkinetic model: CSTR reactor 1,2 DCA (DOI: https://doi.org/10.1021/acscatal.1c00940)   
    ├── evaluate_DRC_fullmodel                 # Degree of rate-control analysis identify sensitive params                    
    ├── full-model                             # Fit parameters with the full order DCA MKM using petBOA and SciPy's Differential Evolution
    ├── inputs                                    
    └── reduced-model                          # Fit parameters with the reduced order DCA MKM using petBOA
```


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

1. It is strongly recommended to install petBOA and all the dependencies in a *new* virtual python enviroment such as [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to make sure all the dependency version requirements can be met and existing work is not disrupted.
2. For instance if you are doing this using conda env these steps should give you the required configuration. 
```
conda create --name petboa python==3.9
conda activate petboa
```
3. Install petboa and the dependencies simply using pip.
   `pip install petboa`
4. Install any missing dependencies using pip.
5. (Optional) If pip and git aren't installed in the conda environment:
```
conda install -c anaconda pip
conda install -c anaconda git
```
6. The examples and the source-code can also be downloaded from GitHub. 
```
https://github.com/VlachosGroup/petBOA.git
```
7. Run the examples in the example folder to look at the four parameter estimation templates that we have created.
8. For instance, if you have cloned the petBOA repository in your $HOME folder.
```
$cd $HOME
$git clone https://github.com/VlachosGroup/petBOA.git
Cloning into 'petBOA'...
remote: Enumerating objects: 1012, done.
remote: Counting objects: 100% (230/230), done.
remote: Compressing objects: 100% (191/191), done.
remote: Total 1012 (delta 56), reused 178 (delta 37), pack-reused 782
Receiving objects: 100% (1012/1012), 16.89 MiB | 31.78 MiB/s, done.
Resolving deltas: 100% (324/324), done.
$cd petBOA 
$cd examples 
$cd example_1_rosenbrock 
$ls
evaluate_GSA petboa       scipy
$cd evaluate_GSA 
$ls
outputs-gsa-salib rosenbrock_GSA.py
$conda activate petboa
(petboa) $python3 rosenbrock_GSA.py 
Para Bounds: [0, 10] [0, 200]
Total number of sensitivity run samples 768
In iteration 1 Loss is 1647.55846 parameters are 9.92 156.59 
In iteration 2 Loss is 1644.39692 parameters are 9.70 156.59 
...
```
9. Every example folder has the Python script necessary to run the example, additional input files, and example output files. 
   1. Note: Please run the Python scripts from their original location in the example folder. 
   2. Note: Examples use relative paths to load input data or models. 
10. Examples 1 and 2 are Python model templates and can be directly used or modified to fit to your purposes.
11. Examples 3 and 4 use OpenMKM based microkinetic models. Therefore, a working version of OpenMKM is necessary to run them.
    1. Instructions to install OpenMKM can be found at: https://github.com/VlachosGroup/openmkm or https://vlachosgroup.github.io/openmkm/
    2. In the example Python scripts, the path to the OpenMKM executable `omkm` path should be modified to suit your system configuration.  
12. If an OpenMKM executable is not present on the machine, then a Docker container for OpenMKM could be used instead.
    1. First install Docker or Docker Desktop directly using instructions from Docker website: https://www.docker.com/get-started/
    2. Make sure the Docker service is started and running before running the petBOA example. 
    3. petBOA uses Docker SDK for Python: https://docker-py.readthedocs.io/en/stable/# to launch and run Docker containers. 
    4. See Example 3: petboa_with_docker for more details. 
    5. Note: In Linux, running Docker requires sudo/admin privileges. Please consult Docker documentation to run Docker from Python in linux. https://docs.docker.com/engine/install/linux-postinstall/ 
13. Please contact us if you have any issues with installation or running the examples. 

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

Article submitted. 
DOI: To-be-added. 