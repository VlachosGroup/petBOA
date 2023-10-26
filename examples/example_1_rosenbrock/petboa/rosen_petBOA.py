"""
Estimate the Rosenbrock function parameters a and b
for the function f(x,y) = (a - x)^2 + b(y - x^2)^2
using generated data similar to a typical parameter
estimation problem using the petBOA optimizer
"""
import os
import time
import numpy as np
import pandas as pd
from numpy import asarray
from petboa.optimizer import BOOptimizer
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


def loss_func(self, params):
    """
    Customized loss function specific to this problem
    """
    a = 1.0
    b = 100.0
    loss = 0.0
    random.seed(7)
    no_exp = 100
    y_model = np.zeros(no_exp)
    y_GT = np.zeros(no_exp)
    for i in range(no_exp):
        x = [random.uniform(-3, 2), random.uniform(-2, 3)]
        x = [random.uniform(-3, 2), random.uniform(-2, 3)]
        y_model[i] = self.model(params[0], params[1], x)
        y_GT[i] = self.model(a, b, x)
    loss = RMSE(y=y_GT, yhat=y_model)
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss])
    self.param_evolution.append([self.call_count] + list(params))
    print("In iteration {} Loss is {:.5f} parameters are {:.2f} {:.2f} ".format(self.call_count, loss, *params))
    return loss


#  2-D rosenbrock function
para_names, parameter_range = parse_param_file("params.xlsx")

# Alternatively set it up manually
# parameter_range = [[0, 10], [0, 200]]
# para_names = ['a', 'b']

print("Para Bounds: {} {}".format(*parameter_range))

estimator_name = 'rosen-petboa'
a = open('output.log', mode='w')
niter = 100
param_res = []
full_df = pd.DataFrame()

for repeat in range(3):
    start_time = time.time()
    a.write("Repeat {} \n".format(repeat))
    a.write("############################ \n")
    a.write("############################ \n")
    print("Repeat {} \n".format(repeat))
    print("############################")
    print("############################")
    param_range = parameter_range.copy()
    # Set up the Model Wrapper to connect to the user-defined Loss function
    ModelWrapper.loss_func = loss_func  # Provide loss function handle to the Model Wrapper Class
    wrapper = ModelWrapper(model_function=rosenbrock,
                           para_names=para_names,
                           name=estimator_name,
                           )
    # Connect the Bayesian Optimizer
    optimizer = BOOptimizer(estimator_name)
    X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func,
                                              param_range,
                                              acq_func="PI",
                                              n_sample_multiplier=5,
                                              n_iter=niter,
                                              log_flag=True)
    a.write("Objective function called {} times \t".format(wrapper.call_count))
    a.write("Parameters are {:.2f} {:.2f} \n".format(*X_opt))
    end_time = time.time()
    a.write("Total Time: {} sec \n".format(end_time - start_time))
    print("Best X", X_opt)
    df1 = pd.DataFrame(data=wrapper.loss_evolution,
                       columns=['Run No', 'Loss'],
                       )
    df1['Repeat'] = repeat
    df2 = pd.DataFrame(data=wrapper.param_evolution,
                       columns=['Run No'] + para_names,
                       )
    df1 = df1.merge(df2, how='inner', on='Run No')
    full_df = pd.concat([full_df, df1])
    param_res.append(X_opt)

os.chdir(estimator_name)
full_df.to_csv("history.csv")
df3 = pd.DataFrame(data=param_res,
                   columns=para_names,
                   )
df3 = df3.T
df3['mean'] = df3.mean(axis=1)
df3['std'] = df3.std(axis=1)
df3.to_csv("parameter_fits.csv")
