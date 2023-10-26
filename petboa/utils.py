"""
Utility functions
"""
import os
import numpy as np
import shutil
import pandas as pd
from pmutt.io.excel import read_excel

# Loss functions
# reference: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
def RMSE(yhat, y):
    return np.sqrt(np.sum((yhat - y) ** 2) / y.size)


def WeightedRMSE(yhat, y, weights):
    return np.sqrt(np.sum((yhat - y) ** 2 * weights) / y.size)


def clear_cache(estimator_name):
    outputPath = os.path.join(os.getcwd(), estimator_name)
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
    os.mkdir(outputPath)


def write_results(estimator_name, start_time, end_time, loss, X, X_true=None):
    OutPutPath = os.path.join(os.getcwd(), estimator_name)
    FilePath = os.path.join(OutPutPath, estimator_name + '.out')
    file = open(FilePath, "w")
    file.write('Parameter estimation takes {:.2f} min \n'.format((end_time - start_time) / 60))
    file.write('Final loss {:.6f} \n'.format(loss))
    file.write('Parameters are {} \n'.format(X))
    if X_true is not None:
        file.write('True Parameters are {} \n'.format(X_true))
    file.close()
    # Also write the results to stdout i.e. use print
    print('Parameter estimation takes {:.2f} min \n'.format((end_time - start_time) / 60))
    print('Final loss {:.6f} \n'.format(loss))
    print('Parameters are {} \n'.format(X))
    if X_true is not None:
        print('True Parameters are {} \n'.format(X_true))


def para_values_to_dict(xi, para_names):
    """Convert parameter values to a dictionary"""
    para_dict = {}
    for name_i, para_i in zip(para_names, xi):
        para_dict.update({name_i: para_i})
    return para_dict


def parse_param_file(file_path):
        try:
            # Attempt to read as an Excel file
            excel_data = read_excel(file_path,
                                    skiprows=0,
                                    header=0,
                                    )
            bounds = get_param_bounds(excel_data)
            names = get_param_names(excel_data)
        except ValueError:
            try:
                # Attempt to read as a CSV file
                csv_data = pd.read_csv(file_path)
                print(csv_data)
                bounds = get_param_bounds(csv_data)
                names = get_param_names(csv_data)
            except pd.errors.EmptyDataError:
                return None  # The file is empty or could not be read
        return names, bounds

def get_param_bounds(data):
    bounds = []
    for r in data:
        try:
            if r["Select"] is True:
                bounds.append([r['LB'], r['UB']])
            else:
                bounds.append([r['Default']])
        except:
            return None
    return bounds

def get_param_names(data):
    names = []
    for r in data:
        try:
            names.append(r["Name"])
        except:
            return None
    return names
