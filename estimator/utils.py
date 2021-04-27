"""
Utility functions
"""

import numpy as np

# Loss functions
# reference: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
def RMSE(yHat, y):
    return np.sqrt(np.sum((yHat - y)**2) / y.size)


def WeightedRMSE(yHat, y, weights):
    return np.sqrt(np.sum((yHat - y)**2 * weights) / y.size)