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



def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    pass


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    pass


def download_data() -> List[Dict[str, Any]]:
    pass
