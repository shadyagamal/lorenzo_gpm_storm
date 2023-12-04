#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:29:52 2023

@author: comi
"""
import numpy as np
import pandas as pd
import somoclu
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.features.routines import get_gpm_storm_patch
from gpm_api.utils.utils_cmap import get_colorbar_settings





def create_map_for_variable_grouped_by_som(df_scaled_prova,df_bmus, variable):
    # Add som node i, j to df 
    merged_df = pd.concat([df_scaled_prova, df_bmus], axis=1)
    
    # Groupby i,j  .apply mean/mean, max, min 
    grouped_df = merged_df.groupby(['Id1', 'Id2'])
    
    
    
    # Calculate the mean for each group and each variable
    mean_df = grouped_df.mean()
    mean_df = grouped_df.mean().reset_index()
    
    
    # Save df_summary
    grid_size = 10
    
    # Create a 2D array for x, y, and color values
    x_values = mean_df['Id2'].values
    y_values = mean_df['Id1'].values
    color_values = mean_df[variable].values
    
    # Create a grid for plotting
    grid = []
    
    # Fill the grid with color values at the corresponding positions
    for x, y, color in zip(x_values, y_values, color_values):
        grid.append([x, y, color])
    
    # Separate x, y, and color values
    grid = np.array(grid)
    x = grid[:, 0]
    y = grid[:, 1]
    color = grid[:, 2]
    
    # Plot the heatmap
    plt.pcolor(x.reshape(grid_size, grid_size), y.reshape(grid_size, grid_size), color.reshape(grid_size, grid_size), cmap='viridis')
    plt.colorbar(label=f'{variable} Mean')
    plt.xlabel('First ID')
    plt.ylabel('Second ID')
    plt.title('Variable Mean Heatmap')
    plt.show()
    
    
# Plot feature statistics map 

# Plot spatial pattern map 

# For single SOM node, plot examples 

# Plot distance between neighbours of some features (avg, distance ...)


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
def add_image(images, i, j, ax):
    plot_kwargs = {} 
    cbar_kwargs = {}

    plot_kwargs, cbar_kwargs = get_colorbar_settings(
        name=images[i, j].name, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs
    )   
    
    max_value_position = np.unravel_index(np.argmax(images[i, j].values), images[i, j].shape)

    # Extract row and column indices
    center_y, center_x = max_value_position
    if center_x < 25:
        img = images[i, j][:, 0:49]
    elif (images[i,j].shape[1] - center_x) > 25:
        start_x = center_x - 24
        end_x = center_x + 25
        img = images[i, j][:, start_x:end_x]
    else: 
        img = images[i, j][:, -49:]
    ax.imshow(img, **plot_kwargs)
    ax.set_title("")  # Set title to an empty string
    ax.set_xlabel("")  # Set xlabel to an empty string
    ax.set_ylabel("")  # Set ylabel to an empty string
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
file_path = '/home/comi/Projects/dataframe.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled_prova = df_scaled.iloc[:40000, :]
df_prova = df.iloc[:40000, :]
# Extract the relevant features from your DataFrame
data = df_scaled.iloc[:40000, :-30].values

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
df_bmus = pd.DataFrame(bmus, columns=['Id1', 'Id2'])

# Save with PICKLE 
# Specify the filename where you want to save the trained SOM
filename = 'som_model_1.pkl'

# Save the trained SOM
with open(filename, 'wb') as file:
    pickle.dump(som, file)
    
# # Load the trained SOM from the file
# with open(filename, 'rb') as file:
#     loaded_som = pickle.load(file)  
    
create_map_for_variable_grouped_by_som(df_scaled_prova, df_bmus, variable='percentage_rainy_pixels_between_10_and_20')

    
# Create images for each cell in the SOM grid
images = create_images_for_som(som, df_prova, bmus)

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
        add_image(images=images, i=i, j=j, ax=ax)
        
    







