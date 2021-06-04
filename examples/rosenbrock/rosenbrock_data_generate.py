"""
Generate data for the Rosenbrock function
to estimate parameters a and b
f(x,y) = (a - x)^2 + b(y - x^2)^2
Data generated using a = 1 and b = 100
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rosenbrock(input_mesh, a, b):
    """
    The Rosenbrock function.
    The function computed is::
        f(x,y) = (a - x)^2 + b(y - x^2)^2
    Parameters
    ----------
    input_mesh : array_like
        2-D Meshgrid at which the Rosenbrock function is to be computed.
    a : float
        Value of parameter a
    b : float
        Value of parameter b
    Returns
    -------
    f : float
        The value of the Rosenbrock function for the grid of x and y
    """
    _x, _y = input_mesh
    return (a - _x) ** 2 + b * (_y - _x ** 2) ** 2


def generate_data(x=np.linspace(-2, 2, 10),
                  y=np.linspace(-1, 3, 10),
                  a=1.00,
                  b=100.00,
                  file_name='data.csv'):
    """
    Generate data used for parameter estimation
    for a rudimentary implementation of Rosenbrock function
    """
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock((X, Y),
                   a=a,
                   b=b)
    fig = plt.figure(0)
    plt.clf()
    plt.contourf(X, Y, Z, 20)
    plt.colorbar()
    plt.contour(X, Y, Z, 20, colors="black")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.show()
    plt.savefig('rosenbrock-contour.png')
    data = np.array([X, Y, Z]).reshape(3, -1).T
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
    df.to_csv('data.csv', index=False)


if __name__ == "__main__":
    generate_data(a=1.00, b=100.00)
