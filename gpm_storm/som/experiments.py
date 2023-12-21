#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:24:28 2023

@author: ghiggi
"""
import pickle
import os 
# TODO: HERE YOU DESCRIBE THE FEATURE YOU USE
# the name is the experiment som_name 

def get_experiment_info(name): 
    info_experiment = {
        "high_intensity_SOM": {
            "features": ["precipitation_max",
                         "count_rainy_areas_over_50",
                         "mean_for_rainy_pixels_over_50",
                         "count_rainy_pixels_over_50",
                         "aspect_ratio_largest_patch_over_20",
                         "echodepth30_mean",
                         "echodepth30_max",
                         "echotopheight50_mean",
                         "echotopheight30_max"],
            "som_grid_size": (10, 10)
           }

    }
    return info_experiment[name]


def filter_nan_values(df, feature):
    """ Filter out rows where a specified variable has NaN values."""
    filtered_df = df.dropna(subset=[feature])
    return filtered_df



def save_som(som, som_dir, som_name):
    # Save SOM
    som_filepath = os.path.join(som_dir, som_name + ".pkl")

    # Save the trained SOM
    with open(som_filepath, 'wb') as file:
        pickle.dump(som, file)
       
        
def load_som(som_dir, som_name):  
    som_filepath = os.path.join(som_dir, som_name + ".pkl")

    with open(som_filepath, 'rb') as file:
        som = pickle.load(file)  
    return som 

