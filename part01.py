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

"""
Create a graph with three subplots displaying the functions f1, f2, and the sum f1+f2 within the range 𝑡 ∈ <0,100>. 
In the third subplot, there will be a section where the value of the sum of both functions exceeds the value of the function f1. 
This section should be shown in green, otherwise in red.  
The conditions for the arguments show_figure and save_path are the same as in the second task.
"""
def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    # Create time values from 0 to 100
    t = np.linspace(0, 100, 1000)

    # Calculate the values of f1, f2, and their sum
    f1_values = 0.5 * np.cos(1/50 * np.pi * t)
    f2_values = 0.25 * (np.sin(np.pi * t) + np.sin(3/2 * np.pi * t))
    sum_values = f1_values + f2_values

    # Create subplots with specific layout
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot f1 in the first subplot
    axes[0].plot(t, f1_values)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('$f_1(t)$')
    axes[0].grid(True)

    # Plot f2 in the second subplot
    axes[1].plot(t, f2_values)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('$f_2(t)$')
    axes[1].grid(True)

    # Plot the sum and highlight regions where the sum is above f1 in green
    plt.fill_between(t, f1_values, sum_values, where=(sum_values > f1_values), color='green')
    plt.fill_between(t, f1_values, sum_values, where=(sum_values <= f1_values), color='red')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('$f_1(t)$ + $f_2(t)$')
    axes[2].grid(True)

    # Customize the x-axis ticks to match the provided example
    for ax in axes:
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels(['0', '20', '40', '60', '80', '100'])

    # Adjust spacing between subplots
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_figure:
        plt.show()

"""
"""
def download_data() -> List[Dict[str, Any]]:
    # URL from which you are downloading the data
    url = "https://ehw.fit.vutbr.cz/izv/stanice.html"
    
    # Download the content of the web page
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find the table on the page
        table = soup.find('table')
        
        # Initialize a list for records
        data = []
        
        # Iterate through the table rows and create records
        for row in table.find_all('tr')[1:]:  # Skip the first row with headers
            columns = row.find_all('td')
            if len(columns) >= 4:
                position = columns[0].text.strip()
                lat = float(columns[1].text.strip())
                long = float(columns[2].text.strip())
                height = float(columns[3].text.strip())
                
                # Create a record
                station_data = {'position': position, 'lat': lat, 'long': long, 'height': height}
                
                # Add the record to the list
                data.append(station_data)
        
        return data
    else:
        print("Error while downloading the page.")
        return []
