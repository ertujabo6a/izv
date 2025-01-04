#!/usr/bin/python3.10
# coding=utf-8
"""
IZV cast3 projektu
Autor: Aleksandr Dmitriev (240259/xdmitr01)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import Normalize
import matplotlib.cm as cm

region_map = {
        0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
        6: "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS",
        17: "PAK", 18: "LBK", 19: "KVK"
    }

def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame from accidents and locations DataFrames. Filters and corrects coordinates.

    Args:
        df_accidents (pd.DataFrame): DataFrame containing accident data.
        df_locations (pd.DataFrame): DataFrame containing location data.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with valid accident locations.
    """
    df_locations = df_locations[(df_locations['d'] != 0) & (df_locations['e'] != 0)]
    swapped = df_locations['d'] < df_locations['e']
    df_locations.loc[swapped, ['d', 'e']] = df_locations.loc[swapped, ['e', 'd']].values

    merged = pd.merge(df_accidents, df_locations, on='p1', how='inner')

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged['d'], merged['e']),
        crs="EPSG:5514"  # S-JTSK
    )

    return gdf

def plot_geo(gdf: gpd.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Plots accidents influenced by alcohol in a selected region for two chosen months.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing accident data.
        fig_location (str): Filepath to save the figure.
        show_figure (bool): Whether to display the figure.
    """
    region_code = 1
    # Filter data for accidents under alcohol influence in the selected region
    alcohol_accidents = gdf[(gdf['p11'] >= 4) & (gdf['p4a'] == region_code)]
    months = [6, 9]
    month_names = {6: "Červenec", 9: "Září"}
    region_name = region_map.get(region_code, "Unknown")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, month in zip(axes, months):
        month_data = alcohol_accidents[alcohol_accidents['p2a'].str.startswith(f'{month:02}')]
        month_data.plot(ax=ax, markersize=15, color='red')
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title(f"{region_name} kraj - pod vlivem alkoholu ({month_names[month]})")
        ax.axis('off')

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_cluster(gdf: gpd.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plots animal-related accidents in a selected region with clusters highlighted as polygons.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing accident data.
        fig_location (str): Filepath to save the figure.
        show_figure (bool): Whether to display the figure.
    """
    region_code = 7
    # Filter data for the selected region and animal-related accidents
    animal_accidents = gdf[(gdf['p10'] == 4) & (gdf['p4a'] == region_code)].copy()

    # Check if there are any incidents in the selected region, because not all regions are represented in our data
    if animal_accidents.empty:
        print(f"Chyba: Pro region {region_code} neexistují žádná data o nehodách zaviněných lesní zvěří.")
        return

    coords = animal_accidents.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    region_name = region_map.get(region_code, "Unknown")

    # Set number of clusters (can be adjusted for clarity and informative visualization).

    # I initially tested DBSCAN as an alternative clustering method,
    # as it is well-suited for spatial data and can handle noise effectively.
    # However, DBSCAN faced challenges in detecting clusters in datasets with uneven density,
    # resulting in either too many clusters or some points being classified as noise and excluded from any cluster.
    #
    # Due to these limitations, I chose Agglomerative Clustering because:
    # - It allows explicit control over the number of clusters (n_clusters), ensuring clearer and more interpretable results.
    # - Unlike DBSCAN, it does not classify points as noise, which is advantageous for visualizations where all accidents should be included.
    # - It generates continuous polygonal areas (convex hulls) that better represents the geographical distribution of data.
    # - The resulting visualizations are more intuitive and easier to interpret.

    n_clusters = 8
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(coords)
    animal_accidents.loc[:, 'cluster'] = clustering.labels_

    # Generate polygons (convex hulls) for clusters
    polygons = []
    cluster_sizes = []
    for cluster_id, cluster_data in animal_accidents.groupby('cluster'):
        if len(cluster_data) > 2:  # Convex hull requires at least 3 points
            hull = cluster_data.geometry.union_all().convex_hull
            polygons.append(hull)
            cluster_sizes.append(len(cluster_data))

    cluster_gdf = gpd.GeoDataFrame({'geometry': polygons, 'size': cluster_sizes}, crs=gdf.crs)
    norm = Normalize(vmin=min(cluster_sizes), vmax=max(cluster_sizes))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(15, 15))

    for _, row in cluster_gdf.iterrows():
        color = cmap(norm(row['size']))
        gpd.GeoSeries([row['geometry']]).plot(ax=ax, color=color, alpha=0.5)

    animal_accidents.plot(ax=ax, color='red', markersize=15, label="Nehody", alpha=0.5)
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label("Počet nehod v úseku")
    ax.set_title(f"Nehody v {region_name} kraji zaviněné lesní zvěří")
    ax.axis('off')
    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    accidents = pd.read_pickle("accidents.pkl.gz")
    locations = pd.read_pickle("locations.pkl.gz")
    geodf = make_geo(accidents, locations)

    plot_geo(geodf, "geo1.png", True)
    plot_cluster(geodf, "geo2.png", True)

    # testovani splneni zadani
    import os
    assert os.path.exists("geo1.png")
    assert os.path.exists("geo2.png")
