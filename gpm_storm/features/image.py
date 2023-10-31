#!/usr/bin/env python3
"""
Created on Tue Oct 31 15:12:57 2023

@author: ghiggi
"""
import cv2
import numpy as np


def _calculate_mean_std_max_stats(data):
    data = data[~np.isnan(data)]

    if data.size == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(data)
    std = np.std(data)
    max_value = np.max(data)

    return mean, std, max_value


def calculate_image_statistics(ds_patch):
    """Define a function to calculate statistics for a patch."""
    # Extract data from ds_patch
    # ds_patch["zFactorFinal"] = ds_patch["zFactorFinal"].compute()
    # ds_patch["precipRateNearSurface"] = ds_patch["precipRateNearSurface"].compute()

    # Create a dictionary to store the dict_results
    dict_results = {}

    # Compute precipitation statistics
    dict_results["precipitation_average"] = np.nanmean(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_std"] = np.nanstd(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_pixel"] = np.nansum(ds_patch["precipRateNearSurface"].data > 0)
    dict_results["precipitation_sum"] = np.nansum(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_max"] = np.nanmax(ds_patch["precipRateNearSurface"].data)

    # Threshold values to iterate over, including 0 for count_over_0
    thresholds = [0, 5, 10, 20, 50]

    # Iterate over thresholds and count_over variables
    for threshold in thresholds:
        # Count patches for the current threshold
        count_patches, _ = cv2.connectedComponents(
            (ds_patch["precipRateNearSurface"].data > threshold).astype(np.uint8), connectivity=8
        )
        dict_results[f"count_patches_over_{threshold}"] = count_patches

        # Count values for the current threshold
        count_over = np.sum(ds_patch["precipRateNearSurface"].data > threshold)
        dict_results[f"count_over_{threshold}"] = count_over

    # Compute reflectivity-related statistics
    ds_patch["REFCH"] = ds_patch.gpm_api.retrieve("REFCH").compute()
    ds_patch["echodepth18"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=18).compute()
    ds_patch["echodepth30"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=30).compute()
    ds_patch["echodepth50"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=50).compute()
    ds_patch["echotopheight18"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=18).compute()
    ds_patch["echotopheight30"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=30).compute()
    ds_patch["echotopheight50"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=50).compute()

    # Calculate mean/std/statistics for each variable with NaN values excluded and store in the dictionary
    variables = [
        "REFCH",
        "echodepth18",
        "echodepth30",
        "echodepth50",
        "echotopheight30",
        "echotopheight50",
    ]
    for variable in variables:
        mean, std, max_value = _calculate_mean_std_max_stats(ds_patch[variable].data)
        dict_results[f"{variable}_mean"] = mean
        dict_results[f"{variable}_std"] = std
        dict_results[f"{variable}_max"] = max_value

    # Add image patch identifiers (image center)
    dict_results["time"] = ds_patch["time"][round(ds_patch["time"].data.shape[0] / 2)].data
    dict_results["lon"] = ds_patch["lon"][round(ds_patch["lon"].data.shape[0] / 2)][
        round(ds_patch["lon"].data.shape[1] / 2)
    ].data
    dict_results["lat"] = ds_patch["lat"][round(ds_patch["lat"].data.shape[0] / 2)][
        round(ds_patch["lat"].data.shape[1] / 2)
    ].data

    return dict_results
