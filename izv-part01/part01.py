#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Aleksandr Dmitriev (240259/xdmitr01)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any

def distance(a: np.array, b: np.array) -> np.array:
    """
    Calculate the Euclidean distance between two arrays of points.

    Args:
        a (np.array): An array of shape (N, D), where N is the number of points and D is the dimension.
        b (np.array): Another array of shape (N, D), representing points in the same space as `a`.

    Returns:
        np.array: An array of distances of shape (N,), where each element represents the Euclidean
                  distance between the corresponding points in `a` and `b`.
    """
    return np.sqrt(np.sum(np.square(a - b), axis=1))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Generate and optionally display or save a graph of sinusoidal functions based on input coefficients.

    Args:
        a (List[float]): A list of coefficients to apply to the sinusoidal function.
        show_figure (bool, optional): If True, displays the generated plot. Defaults to False.
        save_path (str | None, optional): Path to save the plot image. If None, the plot is not saved. Defaults to None.

    Returns:
        None
    """
    x = np.arange(0, 6 * np.pi, 0.1)
    a_values = np.array(a)
    y_values = (a_values[:, np.newaxis] ** 2) * np.sin(x)

    fig = plt.figure(figsize=(10, 4))
    for i, val in enumerate(y_values):
        plt.plot(x, val, label=f'$y_{a[i]}(x)$')
        plt.fill_between(x, val, alpha=0.1)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$f_a(x)$')
    plt.xlim(0, 6 * np.pi)
    xticks_positions = [i * np.pi / 2 for i in range(13)]
    xticks_labels = [
        r'$0$' if i == 0 else
        r'$\pi$' if i == 2 else
        r'${}\pi$'.format(i // 2) if i % 2 == 0 else
        r'$\frac{{{}}}{{2}}\pi$'.format(i) for i in range(13)
    ]
    plt.xticks(xticks_positions, xticks_labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    if save_path:
        plt.savefig(save_path)

    if show_figure:
        plt.show()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generate a multi-plot figure of sinusoidal waveforms and their sum, with dynamic color.

    Args:
        show_figure (bool, optional): If True, displays the generated plot. Defaults to False.
        save_path (str | None, optional): Path to save the plot image. If None, the plot is not saved. Defaults to None.

    Returns:
        None
    """
    t = np.linspace(0, 100, 1000)
    f1 = 0.5 * np.cos(np.pi * t / 50)
    f2 = 0.25 * (np.sin(np.pi * t) + np.sin(1.5 * np.pi * t))
    f_sum = f1 + f2

    fig, axis = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    axis[0].plot(t, f1, label=r'$f_1(t)$')
    axis[0].set_xlim(0, 100)
    axis[0].set_ylim(-0.8, 0.8)
    axis[0].set_ylabel(r'$f_1(t)$')

    axis[1].plot(t, f2, label=r'$f_2(t)$')
    axis[1].set_xlim(0, 100)
    axis[1].set_ylim(-0.8, 0.8)
    axis[1].set_ylabel(r'$f_2(t)$')

    for i in range(len(t) - 1):
        if f_sum[i] > f1[i]:
            color = 'g'
        elif t[i] < 50:
            color = 'r'
        elif t[i] >= 50:
            color = 'orange'
        axis[2].plot(t[i:i + 2], f_sum[i:i + 2], color=color)

    axis[2].set_xlim(0, 100)
    axis[2].set_ylim(-0.8, 0.8)
    axis[2].set_ylabel(r'$f_1(t) + f_2(t)$')

    if save_path:
        plt.savefig(save_path)

    if show_figure:
        plt.show()


def download_data() -> Dict[str, List[Any]]:
    """
    Download and parse weather station data from a webpage, extracting positions, latitudes, longitudes,
    and heights of weather stations.

    Returns:
        Dict[str, List[Any]]: A dictionary with weather station data, containing:
            - 'positions': List of station names or identifiers.
            - 'lats': List of latitudes as floats.
            - 'longs': List of longitudes as floats.
            - 'heights': List of elevations/heights as floats.
    """
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find_all('table')[1]
    weather_station_data = {
        'positions': [],
        'lats': [],
        'longs': [],
        'heights': [],
    }

    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        if len(columns) >= 4:
            position = columns[0].text.strip()
            lat = float(columns[2].text.strip().replace(',', '.').strip('°'))
            long = float(columns[4].text.strip().replace(',', '.').strip('°'))
            height = float(columns[6].text.strip().replace(',', '.').strip('°'))
            weather_station_data['positions'].append(position)
            weather_station_data['lats'].append(lat)
            weather_station_data['longs'].append(long)
            weather_station_data['heights'].append(height)

    return weather_station_data

