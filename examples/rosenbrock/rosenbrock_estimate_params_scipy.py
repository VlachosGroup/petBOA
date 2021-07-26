"""
Estimate the Rosenbrock function parameters a and b
for the function f(x,y) = (a - x)^2 + b(y - x^2)^2
using generated data similar to a typical parameter
estimation problem
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from estimator.modelwrappers import ModelWrapper
from estimator.utils import WeightedRMSE
from rosenbrock_data_generate import generate_data

# Change x,y,a, b to solve a
# new generate data for a
# parameter estimation problem
generate_new_data = False
if (generate_new_data):
    a = 10.0
    b = 200.0
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


# Try SciPy Optimizers for the same task
def loss_func_scipy(x, wrapper):
    """
    Customized loss function specific to this problem
    """
    loss = 0
    #  customization specific to the problem
    _X, _Y = np.meshgrid(wrapper.x_inputs[0],
                         wrapper.x_inputs[1])

    _Z = wrapper.model((_X, _Y), x)
    y_predict = _Z.reshape(1, -1)
    #  end customization specific to the problem
    for i in range(wrapper.n_responses):
        # Factor in the weights
        loss += WeightedRMSE(wrapper.y_groundtruth[i], y_predict[i], wrapper.y_weights)
    wrapper.call_count += 1
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

start_time = time.time()
estimator_name = 'rosenbrock-test-scipy'

wrapper = ModelWrapper(model_function=rosenbrock,  # model function used for evaluating responses = f(inputs,params)
                       para_names=para_names,
                       name=estimator_name,
                       )
wrapper.input_data(x_inputs=x_input,
                   n_trials=100,
                   y_groundtruth=y_response)
bounds = Bounds([0.0, 5.0], [50.0, 150.0])
res = minimize(loss_func_scipy,
               x0=[-50.0, 0.0],
               args=(wrapper,),
               method="Nelder-Mead",
               options={'xtol': 1e-8, 'disp': True},
               bounds=bounds,
               )
end_time = time.time()
a = open("debug-scipy.log",'w')
a.write("Objective function called {} times \n".format(wrapper.call_count))
a.write("Parameters are {} \n".format(res['x']))
a.write("Total time taken in sec {} \n \n \n".format(end_time-start_time))
a.write("Optimizer results {}".format(res))
end_time = time.time()
print(res)
