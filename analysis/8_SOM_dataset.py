#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:29:52 2023

@author: comi
"""
import os
import somoclu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from gpm_storm.som.experiments import get_experiment_info, save_som, load_som
from gpm_storm.features.dataset_analysis import filter_nan_values # TO PUT IN gpm_storm.som.preprocessing !  
from gpm_storm.som.io import (
    sample_node_datasets,
    create_som_sample_ds_array,
    create_som_df_array,
    create_dask_cluster,
)
from gpm_storm.som.plot import (
    plot_images,
    add_image,
    create_map_for_variable_grouped_by_som
)
 
file_path = '/home/comi/Projects/dataframe2.parquet'
som_dir = "/home/comi/Projects/gpm_storm/scripts/" # TODO to change ... 
som_name = "high_intensity_SOM" # TODO: THIS IS THE NAME IDENTIFYING THE EXPERIMENT


# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Define the features to train the SOM 
info_dict = get_experiment_info(som_name)  # HERE INSIDE YOU DEFINE THE EXPERIMENT (features, som_settings ...)
features = info_dict["features"]
n_rows, n_columns = info_dict["som_grid_size"] 

# Subset here the dataframe row to discard (i.e. nan stuffs, select only high intensity ...)
# TODO: TO IMPROVE ! IDEALLY PARAMETRIZE IN get_experiment_info !
# How can you recall the preprocessing ... if you modify manually the code for each trained som !!!! 
# # for feature in features:
#     df = filter_nan_values(df, features)
df = df.dropna(subset=features) # MAYBE THIS IS ENOUGH for the moment ... but are discarding lot of stuffs... to be reported ! 


#####--------------------------------------------------------------------------.
##### SOM TRAINING --> Put this in a single file --> som_training.py  
# Select columns with the relevant dataframe features
df_features = df[features]

# Standardize the dataframe 
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_features))

# Extract the data array
data = df_scaled.to_numpy()

# Initialize the SOM
som = somoclu.Somoclu(n_columns=n_columns, n_rows=n_rows, \
                      gridtype='rectangular', maptype='planar') #  initialcodebook)  ...sample the original codes 

# Train SOM
# train(data=None, epochs=10,  scale0=0.1, scaleN=0.001, scalecooling='linear')
som.train(data=data, epochs=50, \
          radius0=0, radiusN=1, \
          scale0=0.5, scaleN=0.001)

# Save the trained SOM
save_som(som, som_dir=som_dir, som_name=som_name)

#####--------------------------------------------------------------------------.
##### SOM ANALYSIS --> Put this in a single file --> som_analysis.py  
som = load_som(som_dir=som_dir, som_name=som_name)

# Get the Best Matching Units (BMUs) for each data point
bmus = som.bmus

# Add to dataframe 
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
        
    
    
create_map_for_variable_grouped_by_som(df, variable='precipitation_average')



# YOU NEED TO CLEAN THE CODE ... JUST WHAT YOU USE ! 
# PUT EVERYTHING ELSE OR A COPY in /dev ! 

# previous_bmus = som.bmus

# n_changed 
# np.sum(~np.all(som.bmus == previous_bmus, axis=1))


# som.update_data 
# som.view_umatrix()
# som.view_similarity_matrix(data[0:10,:])
# som.view_activation_map(data_vector=data[0:10,:])

# dist_matrix = som.get_surface_state(data=data[[0],:6]).reshape(10,10)
# plt.imshow(dist_matrix)




    


    


 

