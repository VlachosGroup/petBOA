"""
Estimate the Rosenbrock function parameters a and b
for the function f(x,y) = (a - x)^2 + b(y - x^2)^2
using generated data similar to a typical parameter
estimation problem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from estimator.optimizer import BOOptimizer, ModelWrapper
from estimator.utils import WeightedRMSE
import estimator.utils as ut
from rosenbrock_data_generate import generate_data

# Change x,y,a, b to solve a
# new generate data for a
# parameter estimation problem
generate_new_data = False
if (generate_new_data):
    a= 1.0
    b= 100.0
    x=np.linspace(-2, 2, 10)
    y=np.linspace(-1, 3, 10)
    generate_data(a=a, b=b)

def rosenbrock(X, params):
    """
    The Rosenbrock function.
    The function computed is::
        f(x,y) = (a - x)^2 + b(y - x^2)^2
    """
    _x, _y = X
    return (params[0] - _x) ** 2 + params[1] * (_y - _x ** 2) ** 2


# def loss_func(self,
#               inputs,
#               ground_truth,
#               model,
#               params,
#               weights):
#     """Generic loss function"""
#     loss = 0
#     """
#     customization specific to the problem
#     """
#     _X, _Y = np.meshgrid(inputs[0],
#                        inputs[1])
#
#     _Z = model((_X, _Y), params)
#     y_predict = _Z.reshape(1, -1)
#     for i in range(len(ground_truth)):
#         # Factor in the weights
#         loss += WeightedRMSE(ground_truth[i], y_predict[0], weights)
#     return loss

# Read data (X,Y,Z) from the data.csv file which is used for fitting the
# parameters a and b.
# You edit this section for your specific problem
df = pd.read_csv('data.csv')
pivot_df = df.pivot(index='X', columns='Y',
                    values='Z')
y = pivot_df.columns.values
x = pivot_df.index.values
data = df.to_numpy()
x_input = [x, y] #experimental inputs read from the csv file.
y_response = data[:,-1:].T

# Set up the problem
# Change the ranges of a and b
# if you generate new data
# using a different a or b
# this are the bounds within which
# parameters are searched
parameter_range = [[0.0, 5.0],  # for default a
                   [90.0, 120.0]]  # for default b
para_names = ['a', 'b']

# start a timer
start_time = time.time()
estimator_name = 'Rosenbrock-Test'
wrapper = ModelWrapper(model_function=rosenbrock,  # model function used for evaluating responses = f(inputs,params)
                       para_names=para_names,
                       name=estimator_name,
                       )
wrapper.input_data(x_inputs=x_input,
                   n_trials=100,
                   y_groundtruth=y_response)
print(wrapper.loss_func([1.5,100]))
optimizer = BOOptimizer(estimator_name)
n_iter = 300
X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func,
                                          parameter_range,
                                          n_iter,
                                          log_flag=True)
end_time = time.time()
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)
