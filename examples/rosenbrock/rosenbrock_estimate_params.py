"""
Estimate the Rosenbrock function parameters a and b
for the function f(x,y) = (a - x)^2 + b(y - x^2)^2
using generated data similar to a typical parameter
estimation problem
"""

import time

import numpy as np
import pandas as pd

import estimator.utils as ut
from estimator.optimizer import BOOptimizer
from estimator.modelwrappers import ModelWrapper
from estimator.utils import WeightedRMSE
from rosenbrock_data_generate import generate_data

# Change x,y,a, b to solve a
# new generate data for a
# parameter estimation problem
generate_new_data = False
if (generate_new_data):
    a = 1.0
    b = 100.0
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-1, 3, 10)
    generate_data(a=a, b=b)


def rosenbrock(X, params):
    """
    The Rosenbrock function.
    The function computed is::
        f(x,y) = (a - x)^2 + b(y - x^2)^2
    """
    _x, _y = X
    return (params[0] - _x) ** 2 + params[1] * (_y - _x ** 2) ** 2


def loss_func(self,
              params):
    """
    Customized loss function specific to this problem
    """
    loss = 0
    #  customization specific to the problem
    _X, _Y = np.meshgrid(self.x_inputs[0],
                         self.x_inputs[1])

    _Z = self.model((_X, _Y), params)
    y_predict = _Z.reshape(1, -1)
    #  end customization specific to the problem
    for i in range(self.n_responses):
        # Factor in the weights
        loss += WeightedRMSE(self.y_groundtruth[i], y_predict[i], self.y_weights)
    self.call_count += 1
    return loss


# Read data (X,Y,Z) from the data.csv file which is used for fitting the
# parameters a and b.
# You edit this section for your specific problem
df = pd.read_csv('data.csv')
pivot_df = df.pivot(index='X', columns='Y',
                    values='Z')
y = pivot_df.columns.values
x = pivot_df.index.values
data = df.to_numpy()
x_input = [x, y]  # experimental inputs read from the csv file.
y_response = data[:, -1:].T

# Set up the problem

# Change the ranges of a and b if you generate new data if using a different a or b
# these are the bounds within which # parameters are searched
parameter_range = [[0.0, 5.0],  # for default a
                   [50.0, 150.0]]  # for default b
para_names = ['a', 'b']

# start a timer
a = open('debug.log', mode='w')
# for i in range(1,2):
#     start_time = time.time()
#     estimator_name = 'rosenbrock-test'
#     ModelWrapper.loss_func = loss_func  # Provide loss function handle to the Model Wrapper Class
#     wrapper = ModelWrapper(model_function=rosenbrock,  # model function used for evaluating responses = f(inputs,params)
#                            para_names=para_names,
#                            name=estimator_name,
#                            )
#     wrapper.input_data(x_inputs=x_input,
#                        n_trials=100,
#                        y_groundtruth=y_response)
#     optimizer = BOOptimizer(estimator_name)
#     n_iter = 50
#     X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func,
#                                               parameter_range,
#                                               n_sample_multiplier=int(i),
#                                               n_iter=n_iter,
#                                               log_flag=True)
#     a.write("Objective function called {} times \t".format(wrapper.call_count))
#     a.write("Parameters are {} \n".format(X_opt))
#     end_time = time.time()
#     a.write("Total Time: {} sec \n".format(end_time-start_time))

start_time = time.time()
estimator_name = 'rosenbrock-test'
ModelWrapper.loss_func = loss_func  # Provide loss function handle to the Model Wrapper Class
wrapper = ModelWrapper(model_function=rosenbrock,  # model function used for evaluating responses = f(inputs,params)
                       para_names=para_names,
                       name=estimator_name,
                       )
wrapper.input_data(x_inputs=x_input,
                   n_trials=100,
                   y_groundtruth=y_response)
optimizer = BOOptimizer(estimator_name)
n_iter = 50
X_opt, loss_opt, Exp = optimizer.optimize(wrapper.loss_func,
                                          parameter_range,
                                          n_sample_multiplier=5,
                                          n_iter=n_iter,
                                          log_flag=True)
a.write("Objective function called {} times \t".format(wrapper.call_count))
a.write("Parameters are {} \n".format(X_opt))
end_time = time.time()
a.write("Total Time: {} sec \n".format(end_time-start_time))
ut.write_results(estimator_name, start_time, end_time, loss_opt, X_opt)
