"""
Generic Model Wrappers for parameter estimation
"""
import numpy as np


class ModelWrapper:
    """Parameter estimation class"""

    def __init__(self, model_function, para_names, name='estimator_0'):
        """Initialize the parameters"""

        self.trials = []
        self.model = model_function
        self.para_names = para_names
        self.name = name
        self.n_trials = 0
        self.n_inputs = 0
        self.n_responses = 0
        self.x_inputs = []
        self.y_responses = []
        self.y_groundtruth = []
        self.y_weights = []
        self.call_count = 0
        self.param_evolution = []
        self.loss_evolution = []

    def input_data(self,
                   x_inputs,
                   n_trials,
                   y_groundtruth,
                   y_weights=None,
                   ):
        """Initialize n data trials"""
        self.n_trials = n_trials
        # Set the ground truth or experimental values
        self.x_inputs = x_inputs
        self.y_groundtruth = y_groundtruth
        self.n_inputs = len(x_inputs)
        self.n_responses = len(y_groundtruth)
        # Set the weights for each data point
        if y_weights is None:
            y_weights = np.ones((self.n_trials, self.n_responses))
        # normalize the weights
        self.y_weights = y_weights / np.sum(y_weights)

    def loss_func(self, params, **kwargs):
        pass
