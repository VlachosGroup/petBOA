"""
Test the module imports
"""

import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_path)

import numpy as np

import estimator.utils as ut

# Calculate a simple RMSE
y1 = np.zeros(4)
y2 = np.ones(4)

print(ut.RMSE(y1, y2))
