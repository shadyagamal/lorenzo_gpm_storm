#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:29:52 2023

@author: comi
"""
import numpy as np
import pandas as pd
import somoclu
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.features.routines import get_gpm_storm_patch
from gpm_api.utils.utils_cmap import get_colorbar_settings


file_path = '/home/comi/Projects/dataframe.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled_prova = df_scaled.iloc[:5000, :]
df_prova = df.iloc[:5000, :]
# Extract the relevant features from your DataFrame
data = df_scaled.iloc[:5000, :-30].values

# Define the size of the SOM grid
som_grid_size = (10, 10)
n_rows, n_columns = 10, 10


# Initialize the SOM
som = somoclu.Somoclu(n_columns=n_columns, n_rows=n_rows, \
                      gridtype='rectangular', maptype='planar') #  initialcodebook)  ...sample the original codes 
#-----------------------------------------------------------------------------.
# Train SOM
# train(data=None, epochs=10,  scale0=0.1, scaleN=0.001, scalecooling='linear')
som.train(data=data, epochs=10, \
          radius0=0, radiusN=1, \
          scale0=0.5, scaleN=0.001)
# Get the Best Matching Units (BMUs) for each data point
bmus = som.bmus

# Function to create an image for each cell in the SOM grid
def create_images_for_som(som, df, bmus):
    som_shape = som.codebook.shape[:-1]
    images = np.empty(som_shape, dtype=object)
    
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            # Extract images for each cell in the SOM
            indices = np.argwhere((bmus[:, 0] == i) & (bmus[:, 1] == j)).flatten()
            if indices.size > 0:
                # Select a representative image (e.g., the first one)
                index = indices[0]
                print(index)
                variable = "precipRateNearSurface"
                granule_id = df.loc[index, 'gpm_granule_id']
                slice_start = df.loc[index, 'along_track_start']
                slice_end = df.loc[index, 'along_track_end']
                date = df.loc[index, 'time']
            
                ds = get_gpm_storm_patch(
                    granule_id=granule_id,
                    slice_start=slice_start,
                    slice_end=slice_end,
                    date=date,
                    verbose=False,
                    variables=variable,
                )
            
                da = ds[variable]
                # Save the processed image data to the images array
                images[i, j] = da  
    
    return images

# Function to show images
def show_images(images, i, j, ax):
    plot_kwargs = {} 
    cbar_kwargs = {}

    plot_kwargs, cbar_kwargs = get_colorbar_settings(
        name=images[i, j].name, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs
    )   
    
    max_value_position = np.unravel_index(np.argmax(images[i, j].values), images[i, j].shape)

    # Extract row and column indices
    center_x, center_y = max_value_position
    if center_y < 25:
        ax.imshow(images[i, j][:, 0:49], **plot_kwargs)
        ax.set_title("")  # Set title to an empty string
        ax.set_xlabel("")  # Set xlabel to an empty string
        ax.set_ylabel("")  # Set ylabel to an empty string
    else:
        if (images[i,j].shape[1] - center_y) > 25: 
        # Use Matplotlib's imshow directly to display the image
            start_y = center_y - 24
            end_y = center_y + 25
            
            # Use Matplotlib's imshow directly to display the image with the specified region
            ax.imshow(images[i, j][:, start_y:end_y], **plot_kwargs)
        else: 
            ax.imshow(images[i, j][:, -49:], **plot_kwargs)
        ax.set_title("")  # Set title to an empty string
        ax.set_xlabel("")  # Set xlabel to an empty string
        ax.set_ylabel("")  # Set ylabel to an empty string
    
# Create images for each cell in the SOM grid
images = create_images_for_som(som, df_prova, bmus)

# Plot the SOM grid with the corresponding images
plt.figure(figsize=(10, 10))
som_shape = som.codebook.shape[:-1]

nrows = som_shape[0]
ncols = som_shape[1]

fig, axes = plt.subplots(nrows, ncols)
fig.subplots_adjust(0,0,1,1,wspace=0, hspace=0)
for i in range(nrows):
    for j in range(ncols):
        ax = axes[i,j]
        show_images(images=images, i=i, j=j, ax=ax)
        ax.axis('off')
    



plt.subplots_adjust(wspace=0, hspace=0)  # Set width and height space to zero

for i in range(som_shape[0]):
    for j in range(som_shape[1]):
        plt.subplot(som_shape[0], som_shape[1], i * som_shape[1] + j + 1)
        show_images(images=images, i=i, j=j)
        plt.axis('off')

plt.show()



