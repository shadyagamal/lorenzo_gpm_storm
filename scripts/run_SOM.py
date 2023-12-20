#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:46:23 2023

@author: comi
"""

import sys
import numpy as np
import pandas as pd
import somoclu
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.features.SOM import create_map_for_variable_grouped_by_som, create_som_df_array, create_som_sample_ds_array, add_image
from gpm_storm.features.dataset_analysis import filter_nan_values


def train_and_save_som(filename, variables_names):   
    file_path = '/home/comi/Projects/dataframe2.parquet'
    
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)  

    #filter the nan values from df for SOM
    for variable_name in variables_names:
        df = filter_nan_values(df, variable_name)
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Extract the relevant features from your DataFrame
    data = df_scaled.loc[:, variables_names].values
    
    
    
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
        
        
        
    # Save with PICKLE 
    # Specify the filename where you want to save the trained SOM
    filename_complete = filename + ".pkl"
    # Get the Best Matching Units (BMUs) for each data point
    # Save the trained SOM
    with open(filename_complete, 'wb') as file:
        pickle.dump(som, file)
        
    # # # Load the trained SOM from the file
    # with open(filename, 'rb') as file:
    #     som = pickle.load(file)  
    
    
    bmus = som.bmus
    
    # #if you want to assess the stocasticity of the process
    # previous_bmus = som.bmus
    # # n_changed 
    # np.sum(~np.all(som.bmus == previous_bmus, axis=1))
    
    
    # som.update_data 
    som.view_umatrix(filename=f'/home/comi/Projects/gpm_storm/data/umatrix_{filename}.png')
    
    som.view_similarity_matrix(data= data[0:40,:], filename=f'/home/comi/Projects/gpm_storm/data/similarity_matrix_{filename}.png')
    
    som.view_activation_map(data_vector=data[0:1,:], filename=f'/home/comi/Projects/gpm_storm/data/activation_map_{filename}.png')
    
    dist_matrix = som.get_surface_state(data=data[[0],:]).reshape(10,10)
    plt.imshow(dist_matrix)
    plt.savefig(f'/home/comi/Projects/gpm_storm/data/distance_matrix_{filename}.png')
    
    
    
    df['row'] = bmus[:, 0]
    df['col'] = bmus[:, 1]
    
    
    arr_df = create_som_df_array(som=som, df=df)
    arr_ds = create_som_sample_ds_array(arr_df, variables="precipRateNearSurface")
    
    
    
    # #get some more exapmles of plot for one specific node
    # row=0
    # col=8
    
    
    # num_images = 10
    # df_node = arr_df[row, col]
    # list_sample_ds = sample_node_datasets(df_node, num_images=num_images, variables="precipRateNearSurface")
    
    
    # # 
    # variable = "precipRateNearSurface"
    # for ds in list_sample_ds:
    #     ds[variable].gpm_api.plot_image()
    #     plt.show() 
        
        
    
        
        
    #create_map_for_variable_grouped_by_som(df, variable='precipitation_average')
    
        
    
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
    plt.savefig(f'/home/comi/Projects/gpm_storm/data/SOM_plot_{filename}.png')
    
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <filename> <variable_names>")
    else:
        # Extract command-line arguments
        filename_arg = sys.argv[1]
        variables_names_arg = sys.argv[2].split(',')

        # Call the function with the provided arguments
        train_and_save_som(filename_arg, variables_names_arg)
