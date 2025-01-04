#!/usr/bin/env python3.12
# coding=utf-8
"""
IZV cast2 projektu
Autor: Aleksandr Dmitriev (240259/xdmitr01)
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import zipfile
import numpy as np

region_map = {
        0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
        6: "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS",
        17: "PAK", 18: "LBK", 19: "KVK"
    }

def load_data(filename: str, ds: str) -> pd.DataFrame:
    """
    Load accident statistics data from a specified dataset file.

    Args:
        filename (str): Path to the ZIP file containing datasets.
        ds (str): Name of the dataset (e.g., 'nehody' or 'nasledky').

    Returns:
        pd.DataFrame: Loaded and combined data from the specified dataset.
    """
    with zipfile.ZipFile(filename, 'r') as zf:
        dataset_file = [file for file in zf.namelist() if ds in file][0]
        with zf.open(dataset_file) as file:
            df = pd.read_html(file, encoding="cp1250")[0]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parse and clean the data for analysis.

    Args:
        df (pd.DataFrame): Raw DataFrame from `load_data`.
        verbose (bool): If True, print the size of the resulting DataFrame.

    Returns:
        pd.DataFrame: Parsed and cleaned DataFrame.
    """
    df['date'] = pd.to_datetime(df['p2a'], errors='coerce')

    df['region'] = df['p4a'].map(region_map)
    df.drop_duplicates(subset='p1', inplace=True)

    if verbose:
        size_in_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"new_size={size_in_mb:.1f} MB")
    return df

def plot_state(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Plot accident counts by road condition for each region.

    Args:
        df (pd.DataFrame): Parsed DataFrame from `parse_data`.
        fig_location (str): Path to save the figure.
        show_figure (bool): Whether to display the figure.

    Returns:
        None
    """
    road_states = {
        1: 'Stav povrchu vozovky v době něhody: povrch suchý',
        2: 'Stav povrchu vozovky v době něhody: povrch suchý',
        3: 'Stav povrchu vozovky v době něhody: povrch mokrý',
        4: 'Stav povrchu vozovky v době něhody: na vozovce je bláto',
        5: 'Stav povrchu vozovky v době něhody: na vozovce je náledí, ujetý sníh',
        6: 'Stav povrchu vozovky v době něhody: na vozovce je náledí, ujetý sníh'
    }
    df['road_state'] = df['p16'].map(road_states)
    df = df[df['p16'].notna()]
    grouped = df.groupby(['road_state', 'region']).size().reset_index(name='count')
    g = sns.catplot(
        data=grouped,
        x='region',
        y='count',
        col='road_state',
        kind='bar',
        col_wrap=2,
        height=4,
        aspect=1.5,
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Kraj", "Počet nehod")
    g.set_xticklabels(region_map.values())
    g.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_alcohol(df: pd.DataFrame, df_consequences: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Plot accidents involving alcohol and their consequences.

    Args:
        df (pd.DataFrame): Parsed DataFrame with accidents data.
        df_consequences (pd.DataFrame): Parsed DataFrame with consequences data.
        fig_location (str): Path to save the figure.
        show_figure (bool): Whether to display the figure.

    Returns:
        None
    """
    injury_levels = {
        1: "Následky: usmrcení",
        2: "Následky: těžké zranění",
        3: "Následky: lehké zranění",
        4: "Následky: bez zranění"
    }
    df_alcohol = df[df['p11'] >= 3]
    merged_df = pd.merge(df_alcohol, df_consequences, on='p1')
    merged_df['person_affected'] = np.where(merged_df['p59a'] == 1, 'Řidič', 'Spolujezdec')
    merged_df['injury_level'] = merged_df['p59g'].map(injury_levels)
    grouped = merged_df.groupby(['region', 'injury_level', 'person_affected']).size().reset_index(name='count')

    g = sns.catplot(
        data=grouped,
        x='region',
        y='count',
        hue='person_affected',
        col='injury_level',
        kind='bar',
        col_wrap=2,
        height=4,
        aspect=1.5,
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Kraj", "Počet nehod")
    g.legend.set_title("Zraněná osoba")
    g.set_xticklabels(region_map.values())
    g.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_type(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Plot accident types over time for selected regions.

    Args:
        df (pd.DataFrame): Parsed DataFrame from `parse_data`.
        fig_location (str): Path to save the figure.
        show_figure (bool): Whether to display the figure.

    Returns:
        None
    """
    accident_types = {
        1: "s jedoucím nekolejovým vozidlem",
        2: "s vozidlem zaparkovaným, odstaveným",
        3: "z pevnou překážkou",
        4: "s chodcem",
        5: "s lesní zvěři",
        6: "s domácím zviřetem",
        7: "s vlakem",
        8: "s tramvaji"
    }
    df['accident_type'] = df['p6'].map(accident_types)

    selected_regions = ["PHA", "JHM", "MSK", "STC"]
    df = df[df['region'].isin(selected_regions)]
    df = df[df['accident_type'].notna()]
    daily_counts = df.groupby(['region', 'date', 'accident_type']).size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    monthly_counts = daily_counts.set_index('date').groupby(['region', 'accident_type']).resample('ME')[
        'count'].sum().reset_index()
    monthly_counts = monthly_counts[(monthly_counts['date'] >= '2023-01-01') & (monthly_counts['date'] <= '2024-10-01')]

    g = sns.relplot(
        data=monthly_counts,
        x="date",
        y="count",
        hue="accident_type",
        col="region",
        col_wrap=2,
        kind="line",
        height=4,
        aspect=1.5
    )

    g.set_titles("Kraj: {col_name}")
    g.set_axis_labels("Měsíc", "Počet nehod")
    g.legend.set_title("Druhy nehod")
    g.tight_layout()


    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


if __name__ == "__main__":
    df = load_data("data_23_24.zip", "nehody")
    df_consequences = load_data("data_23_24.zip", "nasledky")
    df2 = parse_data(df, True)
    
    plot_state(df2, "01_state.png")
    plot_alcohol(df2, df_consequences, "02_alcohol.png", True)
    plot_type(df2, "03_type.png")
