#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:29:26 2023

@author: comi
"""

import glob
import os
import pandas as pd 
import pyarrow.dataset as ds
import pyarrow as pa
import matplotlib.pyplot as plt
import seaborn as sns




def relative_distribution_of_dataset(df):

    columns = df.columns

    # Calculate the number of rows and columns needed for the subplots
    num_columns = len(columns)-4
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

def boxplot_of_dataset(df):
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


# Rest of your code
dst_dir = "/ltenas8/data/GPM_STORM/features_v1"
list_file = glob.glob(os.path.join(dst_dir, "*", "*", "*", "*.parquet"))


dataset = ds.dataset(list_file[0:100])

table = dataset.to_table()
# Create empty list to store successfully loaded tables
# tables = []

# for file_path in list_file:
#     try:
#         # Try to open the file and convert it to a Table
#         table = ds.dataset(file_path).to_table()
        
#         # If successful, add the table to the list
#         tables.append(table)
#     except FileNotFoundError as e:
#         # Handle the FileNotFoundError (or other relevant exceptions) here
#         print(f"Error opening file {file_path}: {e}")

# Concatenate all successfully loaded tables into one
#final_table = pa.concat_tables(tables)

df = table.to_pandas(types_mapper=pd.ArrowDtype) 


# Get the list of column names in your DataFrame

relative_distribution_of_dataset(df)
boxplot_of_dataset(df)

