# Parameter Estimation - BO

This is an open-source software package to facilitate parameter estimation using 
Bayesian optimization implemented by NEXTorch/BOTorch framework.

Documentation
-------------

To-Do


Developers
----------

-  Yifan Wang (wangyf@udel.edu)
-  Sashank Kasiraju (skasiraj@udel.edu)
-  Gerhard Wittreich (wittregr@udel.edu)

Dependencies
------------

-  Python >= 3.7
-  `NEXTorch` >=  : Used for implementing the parameter estimation with BO optimization framework
-  `PyTorch` >= 1.8: Used for tensor operations with GPU and autograd support
-  `GPyTorch` >= 1.4: Used for training Gaussian Processes
-  `BoTorch` >= 0.4.0: Used for providing Bayesian Optimization framework
-  `Matplotlib`: Used for generating plots
-  `PyDOE2`: Used for constructing experimental designs
-  `Numpy`: Used for vector and matrix operations
-  `Scipy`: Used for curve fitting
-  `Pandas`: Used to import data from Excel or CSV files
-  `openpyxl`: Used by Pandas to import Excel files
-  `pytest`: Used for unit tests


Getting Started
---------------

1. Install the dependencies using using pip. Usually installing nextorch should install all other depedencies::

    `pip install nextorch`

Note: It will be a good idea to install all the dependencies in a virtual python enviroment such as [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to make sure you don't corrupt system python.

For instance if you are doing this using conda env these steps should give you the required configuration. 
```
conda create --name parameterbo python
conda activate parameterbo
```

This step is optional. If pip and git aren't installed in the conda environment:
```
conda install -c anaconda pip
conda install -c anaconda git
```

Then inside that virtual environment run:
```
pip install nextorch
```

2. (Optional Step) Run the unit tests for nextorch using the instructions at https://nextorch.readthedocs.io/en/latest/ to test the NEXTorch installation. 
3. We don't have an installer for this package yet, so simply download the source-code and examples using::
```
git clone https://github.com/VlachosGroup/Parameter-Estimation-BO.git
```
4. Add the `Parameter-Estimation-BO` cloned folder to the `PYTHONPATH`, to be able to call all python modules. 
    - If you are using and IDE like Pycharm, then adding the `Parameter-Estimation-BO` folder as a new project, will automatically add all the project files to the `PYTHONPATH`, so you can go ahead and run any examples within Pycharm. 
      Note: Make sure that within Pycharm the python interpreter is changed to the conda virtual env you just created i.e. `parameterbo` in this example. 
    - If not then you need to manually add the `Parameter-Estimation-BO` folder to your `PYTHONPATH`. For Windows10, you could do this using Environment Variables [see stackoverflow instructions here](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages). Contact Sashank (skasiraj@udel.edu) if you have any further questions.
    - Alternatively, if you are using anaconda python, you can simply run this command within your active `env`, where the `<directory>` is the `Parameter-Estimation-BO` folder you just cloned.: 
    - ```
      conda develop <directory>
      ``` 
      Note: You need to do this from one folder above the cloned folder. 

5. Run the examples in the examples folder to look at the four parameter estmation templates that we have created. 

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

