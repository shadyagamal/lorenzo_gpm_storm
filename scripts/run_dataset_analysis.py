#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:35:21 2023

@author: comi
"""

import pandas as pd
from gpm_storm.features.dataset_analysis import preliminary_dataset_analysis, spacial_analysis, filter_nan_values


dst_dir = "/ltenas8/data/GPM_STORM/features_v2"
df, df_no_nan, df_scaled, results = preliminary_dataset_analysis(dst_dir)


#define a file path to save your dataframe
file_path = '/home/comi/Projects/dataframe2.parquet'

# Save the Parquet file
df.to_parquet(file_path)

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

#read columns name
df.columns
  

spacial_analysis(df, color_variable = "precipitation_average")


df_no_nan_variable = filter_nan_values(df, variable_name="aspect_ratio_largest_patch_over_0")

 


