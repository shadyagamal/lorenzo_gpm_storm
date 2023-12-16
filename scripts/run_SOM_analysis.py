#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:07:47 2023

@author: comi
"""
import numpy as np
import pandas as pd
import somoclu
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import polars as pl
import cartopy.crs as ccrs
from gpm_api.visualization.plot import plot_cartopy_background, plot_colorbar
from gpm_api.bucket.analysis import pl_add_geographic_bins, pl_df_to_xarray




    
file_path = '/home/comi/Projects/dataframe.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# df_scaled_prova = df_scaled.iloc[:40000, :]
# df_prova = df.iloc[:40000, :]
# Extract the relevant features from your DataFrame
data = df_scaled.iloc[:, [0,1,2,3,4,5]].values

# Define the size of the SOM grid
som_grid_size = (10, 10)
n_rows, n_columns = 10, 10


# Initialize the SOM
som = somoclu.Somoclu(n_columns=n_columns, n_rows=n_rows, \
                      gridtype='rectangular', maptype='planar') #  initialcodebook)  ...sample the original codes 
#-----------------------------------------------------------------------------.
# Train SOM
# train(data=None, epochs=10,  scale0=0.1, scaleN=0.001, scalecooling='linear')
som.train(data=data, epochs=50, \
          radius0=0, radiusN=1, \
          scale0=0.5, scaleN=0.001)
# Get the Best Matching Units (BMUs) for each data point
bmus = som.bmus


previous_bmus = som.bmus
# n_changed 
np.sum(~np.all(som.bmus == previous_bmus, axis=1))


# som.update_data 
som.view_umatrix()
som.view_similarity_matrix(data[0:10,:])
som.view_activation_map(data_vector=data[0:10,:])

dist_matrix = som.get_surface_state(data=data[[0],:6]).reshape(10,10)
plt.imshow(dist_matrix)
df['row'] = bmus[:, 0]
df['col'] = bmus[:, 1]


arr_df = create_som_df_array(som=som, df=df)
arr_ds = create_som_sample_ds_array(arr_df, variables="precipRateNearSurface")

row=0
col=8
num_images = 10
df_node = arr_df[row, col]
list_sample_ds = sample_node_datasets(df_node, num_images=num_images, variables="precipRateNearSurface")


# 
variable = "precipRateNearSurface"
for ds in list_sample_ds:
    ds[variable].gpm_api.plot_image()
    plt.show() 
    
    

    
    
create_map_for_variable_grouped_by_som(df, variable='precipitation_average')

    

# Plot the SOM grid with the corresponding images
figsize=(10, 10)
som_shape = som.codebook.shape[:-1]

nrows = som_shape[0]
ncols = som_shape[1]

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
fig.subplots_adjust(0,0,1,1, wspace=0, hspace=0)
for i in range(nrows):
    for j in range(ncols):
        ax = axes[i,j]
        add_image(images=arr_ds, i=i, j=j, ax=ax)
        
    


    
# Save with PICKLE 
# Specify the filename where you want to save the trained SOM
filename = 'som_model_first_5_var.pkl'

# Save the trained SOM
with open(filename, 'wb') as file:
    pickle.dump(som, file)
    
# # Load the trained SOM from the file
with open(filename, 'rb') as file:
    som = pickle.load(file)  

df = pd.concat([arr_df[0,4], arr_df[0,5], arr_df[0,6], arr_df[0,7], arr_df[0,3], arr_df[0,8], arr_df[0,9], arr_df[1,5], arr_df[1,6], arr_df[1,7], arr_df[1,8], arr_df[1,9], arr_df[2,5], arr_df[2,6], arr_df[2,7], arr_df[2,8], arr_df[2,9], arr_df[2,5], arr_df[2,6], arr_df[2,7], arr_df[2,8], arr_df[2,9], arr_df[3,5], arr_df[3,6], arr_df[3,7], arr_df[3,8], arr_df[3,9], arr_df[4,5], arr_df[4,6], arr_df[4,7], arr_df[4,8], arr_df[4,9]], axis = 0)

#%%%##plotting and analysis#####
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


