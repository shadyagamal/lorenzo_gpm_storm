#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:29:26 2023

@author: comi
"""

import glob
import os
import pandas as pd 
import numpy as np
import pyarrow.dataset as ds
import pyarrow as pa
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import seaborn as sns




def _relative_distribution_of_dataset(df):

    columns = df.columns

    # Calculate the number of rows and columns needed for the subplots
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed

    # Create subplots with the desired layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    fig.suptitle("Distributions of Data", y=1.02)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot the relative distributions for each column
    for i, column in enumerate(columns):
        # Check if the column is numeric (excluding the 'time' column)
        if pd.api.types.is_numeric_dtype(df[column]) and column != ('time', 'along_track_start', 'along_track_stop', 'gpm_granule_id'):
            sns.histplot(data=df, x=column, kde=True, ax=axes[i], stat='percent', common_norm=False)
            axes[i].set_title(f"Relative Distribution of {column}")
            axes[i].set_xlabel("Values")
            axes[i].set_ylabel("Relative Frequency (%)")

    # Hide any empty subplots
    for i in range(num_columns, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def _boxplot_of_dataset(df):
        # Get the list of column names in your DataFrame
    columns = df.columns
    
    # Calculate the number of rows and columns needed for the subplots
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed
    
    # Create subplots with the desired layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    fig.suptitle("Boxplots of Data", y=1.02)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Plot boxplots for each column
    for i, column in enumerate(columns):
        # Check if the column is numeric (excluding the 'time' column)
        if pd.api.types.is_numeric_dtype(df[column]) and column != ('time', 'along_track_start', 'along_track_stop', 'gpm_granule_id'):
            sns.boxplot(x=column, data=df, ax=axes[i], showfliers=False)
            axes[i].set_title(f"Boxplot of {column}")
            axes[i].set_xlabel("Values")
    
    # Hide any empty subplots
    for i in range(num_columns, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()
    
def _bivariate_analysis(df, x_variable, y_variable, color_variable):
    
    
    df[x_variable] = pd.to_numeric(df[x_variable], errors='coerce')
    df[y_variable] = pd.to_numeric(df[y_variable], errors='coerce')
    df[color_variable] = pd.to_numeric(df[color_variable], errors='coerce')

    # Convert Arrow columns to Pandas Series
    x_values = np.array(df[x_variable])
    y_values = np.array(df[y_variable])
    color_values = np.array(df[color_variable])
    
    # Handle missing values
    df.dropna(subset=[x_variable, y_variable, color_variable], inplace=True)



    # Set the style of seaborn
    sns.set(style="whitegrid")
    
    # Create a scatter plot with color based on the third variable
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_values, y=y_values, hue=color_values, palette='viridis', edgecolor='w', s=100)
    
    # Add labels and a legend
    plt.title(f'Bivariate Analysis of {x_variable} and {y_variable} (Colored by {color_variable})')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.legend(title=color_variable)
    
    # Show the plot
    plt.show()
    
    bins = 20
    
    # Create a 2D histogram with mean values
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(
        x=x_values,
        y=y_values,
        values=color_values,
        statistic='mean',
        bins=bins
    )
    
    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(statistic.T, cmap='viridis', xticklabels=x_edges, cbar=True)
    
 
    # Add labels and a title
    plt.title(f'Bivariate Analysis with Mean Values (Color Coded for {color_variable})')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    
    # Show the plot
    plt.show()



def preliminary_dataset_analysis(dst_dir):
    list_files = glob.glob(os.path.join(dst_dir, "*", "*", "*", "*.parquet"))
    dataset = ds.dataset(list_files)
    
    table = dataset.to_table()

        
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    
    
    # Get the list of column names in your DataFrame
    
    _relative_distribution_of_dataset(df)
    _boxplot_of_dataset(df)
    _bivariate_analysis(df, 'precipitation_pixel', 'precipitation_sum', 'lat')
    return list_files

dst_dir = "/ltenas8/data/GPM_STORM/features_v1"
list_files = preliminary_dataset_analysis(dst_dir)


