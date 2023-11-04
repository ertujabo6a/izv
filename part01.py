#!/usr/bin/env python3
"""
IZV First Part of the Project
Autor: Aleksandr Dmitriev (240259/xdmitr01)
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any

"""
The aim of this function is to numerically calculate the integral using the approximate so-called rectangle method. 
You will receive, as input, a function `f` 
(compatible with NumPy, taking one NumPy array as input and returning a NumPy array of the same size; no testing is required), 
the start and end of the interval `a` and `b`, and an optional number of steps `steps`. If not set, `steps` will default to 1000. 
The function will return the value of the definite integral calculated according to the rectangle method formula.
"""
def integrate(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, steps=1000) -> float:
    # Generate points on a linear distribution from 'a' to 'b'
    x_values = np.linspace(a, b, steps)

    # Calculate the function values at all points
    y_values = f(x_values)

    # Width of one rectangle (step)
    dx = (b - a) / steps

    # Calculate the integral using the rectangle method
    integral_value = np.sum(y_values) * dx

    return integral_value

"""
The aim of generate_graph is to vizualise graph of the function f_a(x) = a^2 * x^3 * sin(x) defined in the range <-3, 3>. 
Generate the function values for all values of 'a' specified in the input argument (represented as a list of floating-point numbers) 
without using loops (i.e., using broadcasting). Then, visualize this matrix row by row to create the following graph. 
For setting display ranges, labels, and similar parameters, you can assume that a = [1.0, 1.5, 2.0]. 
Ensure the LaTeX style for axis labels and individual lines. Calculate the integral values using the trapz function in numpy.
The generate_graph function has two additional arguments - a boolean value 'show_figure'
that determines whether the graph should be displayed using the show() function and 'save_path' 
which (if specified) determines where the graph should be saved using the savefig() function.
"""
def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    # 'a' values fo calculation
    a_values = np.array([1.0, 1.5, 2.0])

    # Range of x
    x = np.linspace(-3, 3, 1000)

    # Calculate values of f(a, x) for all combinations of 'a' and 'x'
    X, A = np.meshgrid(x, a_values)
    result_matrix = A**2 * X**3 * np.sin(X)

    # Calculate integrals for each row of the matrix
    integrals = np.trapz(result_matrix, x, axis=1)

    # Create the graph
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, a in enumerate(a_values):
        ax.plot(x, result_matrix[i], label=f'$γ_{a}(x)$')
        ax.fill_between(x, result_matrix[i], alpha=0.1)
        
        # Add text to the right of the end of the each graph
        ax.annotate(f'$\\int f_{a:}(x)dx= {integrals[i]:.2f}$', xy=(x[-1], result_matrix[i][-1]))

    # Set axis labels, legend, and graph title
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f_a(x)$')
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    if show_figure:
        plt.show()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    pass


def download_data() -> List[Dict[str, Any]]:
    pass
