#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:35:21 2023

@author: comi
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from gpm_storm.features.dataset_analysis import preliminary_dataset_analysis, spacial_analysis, filter_nan_values
import cartopy.crs as ccrs
from gpm_api.visualization.plot import plot_cartopy_background, plot_colorbar
from gpm_api.bucket.analysis import pl_add_geographic_bins, pl_df_to_xarray





dst_dir = "/ltenas8/data/GPM_STORM/features_v2"
df, df_no_nan, df_scaled, results = preliminary_dataset_analysis(dst_dir)



file_path = '/home/comi/Projects/dataframe2.parquet'

# Save the Parquet
df.to_parquet(file_path)

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)
df.columns
  

spacial_analysis(df, color_variable = "precipitation_average")


df_no_nan_variable = filter_nan_values(df, variable_name="aspect_ratio_largest_patch_over_{threshold}")

 
df["lenght_track"] = df["along_track_end"] - df["along_track_start"]


df_rounded = df.copy() 
df_rounded["lon_bin"] = df_rounded["lon"].round(1)
df_rounded["lat_bin"] = df_rounded["lat"].round(1)
grouped_df = df_rounded.groupby(["lon_bin", "lat_bin"])
binned_df = grouped_df.agg(["count", "median"])

df.dtypes
xbin_column="lon_bin"
ybin_column="lat_bin"
bin_spacing=0.1
bin_spacing=2

df["row-col"] = df["col"].astype(str) + "-" + df["row"].astype(str)
df_pl = pl.from_pandas(df)
df_pl = pl_add_geographic_bins(df_pl, xbin_column=xbin_column, ybin_column=ybin_column, 
                               bin_spacing=bin_spacing, x_column="lon", y_column="lat")

grouped_df = df_pl.groupby([xbin_column, ybin_column])
df_stats_pl = grouped_df.agg(pl.col("precipitation_average").count().alias("bin_count"),
                             pl.col("row-col").mode().alias("more_frequent_node")
                             )

ds = pl_df_to_xarray(df_stats_pl,  
                     xbin_column=xbin_column, 
                     ybin_column=ybin_column, 
                     bin_spacing=bin_spacing)


df_subset = df[np.logical_and(df["row"] == 0, df["col"] == 8)]
lon = df_subset["lon"].values
lat = df_subset["lat"].values
value = df_subset["echodepth30_mean"]

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c="orange", s=2)

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
p = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=value, s=4, cmap="Spectral", vmax=5000)
plot_colorbar(p=p, ax=ax)


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
p = ds["bin_count"].plot.imshow(ax=ax, x="longitude", y="latitude", cmap="Spectral", add_colorbar=False)
plot_colorbar(p=p, ax=ax)


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
p = ds["row-col"].plot.imshow(ax=ax, x="longitude", y="latitude", cmap="Spectral", add_colorbar=False)
plot_colorbar(p=p, ax=ax)
