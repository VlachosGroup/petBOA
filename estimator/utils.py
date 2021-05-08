"""
Utility functions
"""
import os 
import numpy as np

# Loss functions
# reference: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
def RMSE(yHat, y):
    return np.sqrt(np.sum((yHat - y)**2) / y.size)


def WeightedRMSE(yHat, y, weights):
    return np.sqrt(np.sum((yHat - y)**2 * weights) / y.size)


def clear_cache(estimator_name):
    OutPutPath = os.path.join(os.getcwd(), estimator_name)
    if os.path.exists(OutPutPath): os.remove(OutPutPath)
    os.mkdir(OutPutPath)

def write_results(estimator_name, start_time, end_time, loss, X):
    
    OutPutPath = os.path.join(os.getcwd(), estimator_name)
    FilePath = os.path.join(OutPutPath, estimator_name + '.out')
    file = open(FilePath,"w")
    file.write('Parameter estimation takes {:.2f} min \n'.format((end_time-start_time)/60))
    file.write('Final loss {:.6f} \n'.format(loss))
    file.write('Parameters are {} \n'.format(X))
    file.close()
