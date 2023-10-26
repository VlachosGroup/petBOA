"""
Estimate the Rosenbrock function parameters a and b
for the function f(x,y) = (a - x)^2 + b(y - x^2)^2
using generated data similar to a typical parameter
estimation problem using the SciPy minimize method.
"""

import time
import numpy as np
import pandas as pd
from numpy import asarray
from scipy.optimize import minimize
from petboa.modelwrappers import ModelWrapper
from petboa.utils import RMSE, parse_param_file
import random


def rosenbrock(a, b, x):
    """
    The Rosenbrock function.
    The function computed is::
        f(x,y) = (a - x)^2 + b(y - x^2)^2
    """
    x = asarray(x)
    _sum = np.sum(b * (x[1:] - x[:-1]**2.0)**2.0 + (a - x[:-1])**2.0,
                  axis=0)
    return _sum


def loss_func(params, _wrapper):
    """
    Customized loss function specific to this problem
    """
    a = 1.0
    b = 100.0
    random.seed(7)
    no_exp = 100
    y_model = np.zeros(no_exp)
    y_GT = np.zeros(no_exp)
    for i in range(no_exp):
        x = [random.uniform(-3, 2), random.uniform(-2, 3)]
        y_model[i] = _wrapper.model(params[0], params[1], x)
        y_GT[i] = _wrapper.model(a, b, x)
    loss = RMSE(y=y_GT, yhat=y_model)
    _wrapper.call_count += 1
    _wrapper.loss_evolution.append([_wrapper.call_count, loss])
    _wrapper.param_evolution.append([_wrapper.call_count] + list(params))
    print("In iteration {} Loss is {:.5f} parameters are {:.2f} {:.2f} ".format(_wrapper.call_count, loss, *params))
    return loss


#  2-D rosenbrock function
para_names, parameter_range = parse_param_file("params.xlsx")

# Alternatively set it up manually
# parameter_range = [[0, 10], [0, 200]]
# para_names = ['a', 'b']

print("Para Bounds: {} {}".format(*parameter_range))
a = open('output.log', mode='w')
niter = 100
param_res = []
full_df = pd.DataFrame()

reps = 2
x_guess = [np.array([random.uniform(0, 10), random.uniform(0, 200)]) for _ in range(reps)]
for repeat in range(reps):
    # start a timer
    start_time = time.time()
    a.write("Repeat {} \n".format(repeat))
    a.write("############################ \n")
    a.write("############################ \n")
    print("Repeat {} \n".format(repeat))
    print("############################ \n")
    print("############################ \n")
    param_range = parameter_range.copy()
    # Set up the Model Wrapper to connect to the user-defined Loss function
    estimator_name = 'rosenbrock-scipy'
    ModelWrapper.loss_func = loss_func  # Provide loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=rosenbrock,
                           para_names=para_names,
                           name=estimator_name,
                           )
    # Give the optimizer an initial guess
    x0 = x_guess[repeat]
    # Call the scipy minimize optimizer
    res = minimize(loss_func,
                   x0=x0,
                   args=(wrapper,),
                   method="Nelder-Mead",
                   tol=1.0E-3,
                   bounds=param_range,
                   options=({'disp': True,
                             'maxiter': niter
                             })
                   )

    # Print the results
    a.write("Parameters are {} loss is {} \n".format(res['x'], res.fun))
    a.write("Objective function called {} times \t".format(wrapper.call_count))
    end_time = time.time()
    a.write("Total Time: {} sec \n".format(end_time - start_time))
    print("Best X", res['x'])
    df1 = pd.DataFrame(data=wrapper.loss_evolution,
                       columns=['Run No', 'Loss'],
                       )
    df1['Repeat'] = repeat
    df2 = pd.DataFrame(data=wrapper.param_evolution,
                       columns=['Run No'] + para_names,
                       )
    df1 = df1.merge(df2, how='inner', on='Run No')
    full_df = pd.concat([full_df, df1])
    param_res.append(res['x'])

# Save the progress to a CSV file for future analysis
full_df.to_csv("history.csv")
df3 = pd.DataFrame(data=param_res,
                   columns=para_names,
                   )
df3 = df3.T
df3['mean'] = df3.mean(axis=1)
df3['std'] = df3.std(axis=1)
df3.to_csv("parameter_fits.csv")
