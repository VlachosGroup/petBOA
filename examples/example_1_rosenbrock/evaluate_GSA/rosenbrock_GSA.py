"""
Estimate the Global sensitivity coefficients
for the loss function of the Rosenbrock
function f(x,y) = (a - x)^2 + b(y - x^2)^2
based parameter estimation problem for the
fitted parameters a and b
"""

import os
import random
import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze
from SALib.sample import sobol
from numpy import asarray
from petboa.modelwrappers import ModelWrapper
from petboa.utils import RMSE


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
    random.seed(7)
    no_exp = 100
    y_model = np.zeros(no_exp)
    y_GT = np.zeros(no_exp)
    for _i in range(no_exp):
        x = [random.uniform(-3, 2), random.uniform(-2, 3)]
        y_model[_i] = self.model(params[0], params[1], x)
        y_GT[_i] = self.model(a, b, x)
    loss = RMSE(y=y_GT, yhat=y_model)
    self.call_count += 1
    self.loss_evolution.append([self.call_count, loss])
    self.param_evolution.append([self.call_count] + list(params))
    print("In iteration {} Loss is {:.5f} parameters are {:.2f} {:.2f} ".format(self.call_count, loss, *params))
    return loss


#  2-D rosenbrock function
parameter_range = [[0, 10], [0, 200]]
print("Para Bounds: {} {}".format(*parameter_range))
para_names = ['a', 'b']
estimator_name = 'outputs-gsa-salib'
if not os.path.exists(estimator_name):
    os.mkdir(estimator_name)
os.chdir(estimator_name)
a = open('output.log', mode='w')
a.write("Para Bounds: {} {} \n".format(*parameter_range))
niter = 100

problem = {
    'num_vars': len(para_names),
    'names': para_names,
    'bounds': parameter_range
}
param_values = sobol.sample(problem, 128)
print("Total number of sensitivity run samples {}".format(param_values.shape[0]))
a.write("Total number of sensitivity run samples {} \n".format(param_values.shape[0]))

Y = np.zeros([param_values.shape[0]])
ModelWrapper.loss_func = loss_func  # Provide loss function handle to the Model Wrapper Class
wrapper = ModelWrapper(model_function=rosenbrock,
                       para_names=para_names,
                       name=estimator_name,
                       )

for i, X in enumerate(param_values):
    Y[i] = wrapper.loss_func(params=X)
    a.write("Finished evaluating {} Sobol sample {:.2f} {:.2f} with loss {:.5f} \n".format(i, *X, Y[i]))


history = np.append(param_values, np.reshape(Y, (len(Y), 1)), axis=1)
hist_df = pd.DataFrame(data=history,
                       columns=["a", "b", "loss"])
hist_df.to_csv("sobol_sample.csv")
si = analyze(problem, Y)
a.write("\n \nFirst Order Sensitivity Indices are:\n {} \n".format(si['S1']))
a.write("Total Order Sensitivity Indices are:\n {} \n".format(si['ST']))

data = {"ST": pd.Series(data=si['ST'],
                        index=para_names),
        "S1": pd.Series(data=si['S1'],
                        index=para_names)}

df = pd.DataFrame(data=data)
print(df)
df.to_csv('results.csv')
