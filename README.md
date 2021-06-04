# Parameter Estimation - BO

This is an open-source software package to facilitate parameter estimation using 
Bayersion optimization implemented by NEXTorch/BOTorch framework.

Documentation
-------------

To-Do


Developers
----------

-  Yifan Wang (wangyf@udel.edu)
-  Sashank Kasiraju (skasiraj@udel.edu)
-  Gerhard Wittreich (wittregr@udel.edu)
-  Max Cohen (maxrc@udel.edu)

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

1. We don't have an installer for this package yet, so simply download the source-code and examples using::
   
   `git clone https://github.com/VlachosGroup/Parameter-Estimation-BO.git`

2. Install the dependencies using using pip. Usually installing nextorch should install all other depedencies::

    `pip install nextorch`
Note: It will be a good idea to install all the dependencies in a virtual python enviroment such as [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to make sure you don't corrupt system python. 

3. Run the unit tests for nextorch using the instructions at https://nextorch.readthedocs.io/en/latest/

4. Run and edit the example templates as needed. 


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

 

Publications
------------

