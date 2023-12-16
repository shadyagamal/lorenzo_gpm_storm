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
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.features.routines import get_gpm_storm_patch
from gpm_api.utils.utils_cmap import get_colorbar_settings




def _add_images_to_subplot(image, output_directory):
    """
    Add images to a subplot.

    Parameters:
    - images: List of images to be plotted.
    - ax: Matplotlib subplot to add images to.
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

        max_value_position = np.unravel_index(np.argmax(image), image.shape)

        # Extract row and column indices
        center_y, center_x = max_value_position
        if center_x < 25:
            img = image[:, 0:49]
        elif (image.shape[1] - center_x) > 25:
            start_x = center_x - 24
            end_x = center_x + 25
            img = image[:, start_x:end_x]
        else: 
            img = image[:, -49:]
            
        plt.imshow(img, cmap='viridis')  # Adjust cmap as needed
        plt.title("")  # Set title to an empty string
        plt.xlabel("")  # Set xlabel to an empty string
        plt.ylabel("")  # Set ylabel to an empty string
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        plt.savefig(os.path.join(output_directory, f"image_{i}_{j}.png"))
        plt.clf()  # Clear the figure for the next image


def _get_node_dataframe(df, row, col):
    """Retrieve feature dataframe of specific SOM node."""
    df_node = df[(df['row'] == row) & (df['col'] == col)]
    return df_node


def _open_sample_dataset(df, index, variables="precipRateNearSurface"):
    granule_id = df.iloc[index]['gpm_granule_id']
    slice_start = df.iloc[index]['along_track_start']
    slice_end = df.iloc[index]['along_track_end']
    date = df.iloc[index]['time']
    ds = get_gpm_storm_patch(
        granule_id=granule_id,
        slice_start=slice_start,
        slice_end=slice_end,
        date=date,
        verbose=False,
        variables=variables,
    )
    return ds


def create_map_for_variable_grouped_by_som(df_final, variable):
    # Groupby i,j  .apply mean/mean, max, min 
    grouped_df = df_final.groupby(['row', 'col'])
    
    # Calculate the mean for each group and each variable
    mean_df = grouped_df.mean()
    mean_df = grouped_df.mean().reset_index()
    
    # Save df_summary
    grid_size = 10
    
    # Create a 2D array for x, y, and color values
    x_values = mean_df['col'].values
    y_values = mean_df['row'].values
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
    plt.gca().invert_yaxis()
    plt.pcolor(x.reshape(grid_size, grid_size), y.reshape(grid_size, grid_size), color.reshape(grid_size, grid_size), cmap='viridis')
    plt.colorbar(label=f'{variable} Mean')
    plt.xlabel('First ID')
    plt.ylabel('Second ID')
    plt.title('Variable Mean Heatmap')
    plt.show()
    
    

def create_som_df_array(som, df):
    """Create SOM array with node dataframes."""
    som_shape = som.codebook.shape[:-1]
    arr_df = np.empty(som_shape, dtype=object)
    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            df_node = _get_node_dataframe(df, row=row, col=col)
            arr_df[row, col] = df_node
    return arr_df 


# Function to create an image for each cell in the SOM grid
def create_som_sample_ds_array(arr_df, variables="precipRateNearSurface"):
    """Open a sample GPM patch dataset for each SOM node."""
    som_shape = arr_df.shape
    arr_ds = np.empty(som_shape, dtype=object)

    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            # Extract images for each cell in the SOM
            df_node = arr_df[row, col]
            # Select valid random index
            index = random.randint(0, len(df_node) - 1)
            # Open dataset
            ds = _open_sample_dataset(df_node, index=index, variables=variables)
            # Add the dataset to the arrays
            arr_ds[row, col] = ds
    return arr_ds


def sample_node_datasets(df_node, num_images=20, variables="precipRateNearSurface"):
    # Limit the number of images to extract
    random_indices = random.sample(range(len(df_node)), num_images)
    list_ds = []
    for index in random_indices:
        print(index)
        ds = _open_sample_dataset(df_node, index=index, variables=variables)
        list_ds.append(ds)
    return list_ds





def _remove_axis(ax): 
    ax.set_title("")  # Set title to an empty string
    ax.set_xlabel("")  # Set xlabel to an empty string
    ax.set_ylabel("")  # Set ylabel to an empty string
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

# Function to show images
def add_image(images, i, j, ax, variable = "precipRateNearSurface"):
    plot_kwargs = {} 
    cbar_kwargs = {}
    
    images[i,j] = images[i,j][variable]
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
    _remove_axis(ax)
    
    
def plot_images(image_list, output_directory):
    num_images = len(image_list)

    # Calculate the number of rows and columns for the subplot grid
    num_rows = int(np.ceil(num_images / 3))  # Adjust as needed
    num_cols = min(num_images, 3)

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))


    for i, ax in enumerate(axes):
        if i < num_images:
            print(image_list[i].data)
            _add_images_to_subplot(image_list[i].data, output_directory)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


 
    
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



