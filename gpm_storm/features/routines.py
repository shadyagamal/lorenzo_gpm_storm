#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:28:29 2023

@author: ghiggi
"""
"""
Spyder Editor

This is a temporary script file.
"""
import os 
import gpm_api
import numpy as np
import pandas as pd
import ximage  # noqa
from gpm_api.io.local import get_time_tree
from gpm_api.io.checks import check_date, check_time
from gpm_api.io.info import get_start_time_from_filepaths
from gpm_storm.features.image import calculate_image_statistics
 

def run_granule_feature_extraction(filepath, dst_dir):
    
    # List some variables of interest
    variables = [
        "sunLocalTime",
        "airTemperature",
        "precipRate",
        "paramDSD",
        "zFactorFinal",
        "zFactorMeasured",
        "precipRateNearSurface",
        "precipRateESurface",
        "precipRateESurface2",
        "zFactorFinalESurface",
        "zFactorFinalNearSurface",
        "heightZeroDeg",
        "binEchoBottom",
        "landSurfaceType",
    ]
    
    # Open granule dataset
    ds = gpm_api.open_granule(filepath, variables=variables, scan_mode="FS")
    
    # %%
    ###################
    #### Labelling ####
    ###################
    min_value_threshold = 0.05
    max_value_threshold = np.inf
    min_area_threshold = 5
    max_area_threshold = np.inf
    footprint = 5
    sort_by = "area"
    sort_decreasing = True
    label_name = "label"
    da = ds["precipRateNearSurface"].compute()
    
    # Retrieve labeled xarray object
    xr_obj = da.ximage.label(
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
        label_name=label_name,
    )
       
    ##################################
    #### Label Patches Extraction ####
    ##################################
    patch_size = (49, 20)
    variable = "precipRateNearSurface"
    # Output Options
    label_name = "label"
    labels_id = None
    n_labels = None
    n_patches = None
    # Patch Extraction Options
    centered_on = "label_bbox"
    padding = 0
    # Define the patch generator
    label_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
        label_name=label_name,
        patch_size=patch_size,
        variable=variable,
        # Output options
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        # Patch extraction Options
        padding=padding,
        centered_on=centered_on,
        # Tiling/Sliding Options
        partitioning_method=None,
    )
        
    # %% patch statistics extraction
    
    n_patches = len(label_isel_dict)
    
    # Calculate statistics for the patch
    # - Read first in memory to speed up computations
    ds["zFactorFinal"] = ds["zFactorFinal"].compute()
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    ds["sunLocalTime"] = ds["sunLocalTime"].compute()
    
    patch_statistics = [
        calculate_image_statistics(ds.isel(label_isel_dict[i][0]), label_isel_dict[i][0]['along_track']) for i in range(1, n_patches)
    ]
        
    # Create a pandas DataFrame from the list
    df = pd.DataFrame(patch_statistics)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f')
    
    # Define filepath 
    start_time = get_start_time_from_filepaths(filepath)
    filename = f"dpr_feature_granule_{filepath}.parquet"
    dirtree = get_time_tree(check_date(check_time(start_time)))
    dir_path = os.path.join(dst_dir, dirtree)
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    
    # Save DataFrame to Parquet
    df.to_parquet(filepath)
