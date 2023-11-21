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
import dask
import gpm_api
import numpy as np
import pandas as pd
import ximage  # noqa
from gpm_api.io.local import get_time_tree, get_local_daily_filepaths
from gpm_api.io.checks import check_date, check_time
from gpm_api.io.info import get_start_time_from_filepaths, get_granule_from_filepaths
from gpm_storm.features.image import calculate_image_statistics



@dask.delayed
def run_feature_extraction(filepath, dst_dir, force):
    with dask.config.set(scheduler="single-threaded"):
        try: 
            run_granule_feature_extraction(filepath, dst_dir=dst_dir, force=force)
            msg = ""
        except Exception as e: 
            msg = f"Processing of {filepath} failed with '{e}'."
    return msg 


def run_granule_feature_extraction(filepath, dst_dir, force=False):
    
    # Define filepath 
    start_time = get_start_time_from_filepaths(filepath)[0]
    filename = os.path.basename(filepath).replace(".HDF5", "")
    filename = f"GPM_STORM.{filename}.parquet"
    dirtree = get_time_tree(check_date(check_time(start_time)))
    dir_path = os.path.join(dst_dir, dirtree)
    os.makedirs(dir_path, exist_ok=True)
    df_filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(df_filepath): 
        if force: 
            os.remove(df_filepath)
        else: 
            raise ValueError(f"force=False and {filepath} already exists.")

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
    
    # Put in memory data for label definition 
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    da = ds["precipRateNearSurface"]
    
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
    n_patches = np.Inf
    # Patch Extraction Options
    centered_on = "label_bbox"
    padding = 0
    # Define the patch generator
    patch_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
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
        
    # Read first in memory to speed up computations [9 seconds]
    ds["zFactorFinal"] = ds["zFactorFinal"].compute()
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    ds["sunLocalTime"] = ds["sunLocalTime"].compute()
    
    # Compute statistics for each patch
    n_patches = len(patch_isel_dict)
    patch_statistics = [
        calculate_image_statistics(ds, patch_isel_dict[i][0]) for i in range(1, n_patches)
    ]
        
    # Create a pandas DataFrame from the list
    df = pd.DataFrame(patch_statistics)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f')
    # Save DataFrame to Parquet
    df.to_parquet(df_filepath)

def patch_plot_and_extraction(granule_id, slice_start, slice_end, date):
    
    date_conv = date.dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    product = "2A-DPR"  # 2A-PR
    filepaths = get_local_daily_filepaths(product, date_conv, product_type="RS", version = 7)
    
    for filepath in filepaths:
        if granule_id == get_granule_from_filepaths(filepath):
            real_filepath = filepath
            break
        
    # List some variables of interest
    variable = ["precipRateNearSurface"]
    
    # Open granule dataset
    ds = gpm_api.open_granule(real_filepath, variables=variable, scan_mode="FS")
    ds = ds.isel(along_track=slice(slice_start, slice_end))
    ds[variable].isel(along_track=slice(slice_start, slice_end)).gpm_api.plot_map()    


    return ds