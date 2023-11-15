#!/usr/bin/env python3

import cv2
import numpy as np
from skimage.measure import regionprops


def _calculate_mean_std_max_stats(data):
    data = data[~np.isnan(data)]

    if data.size == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(data)
    std = np.std(data)
    max_value = np.max(data)

    return mean, std, max_value


def calculate_image_statistics(ds_patch, along_track_slice):
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
    dict_results["center_precipitation_pixel"] = np.nansum(ds_patch["precipRateNearSurface"][10:-10, 10:-10].data > 0)
    dict_results["precipitation_sum"] = np.nansum(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_max"] = np.nanmax(ds_patch["precipRateNearSurface"].data)

    # Threshold values to iterate over, including 0 for count_over_0
    thresholds = [0, 5, 10, 20, 50]
    intensity_ranges = [(0, 1), (1, 5), (5, 10), (10, 20)]

    # Iterate over thresholds and count_over variables
    for threshold in thresholds:
        # Count patches for the current threshold
        count_patches, labeled_image = cv2.connectedComponents(
            (ds_patch["precipRateNearSurface"].data > threshold).astype(np.uint8), connectivity=8
        )
        dict_results[f"count_patches_over_{threshold}"] = count_patches
        if threshold == 0:
            regions = regionprops(labeled_image)
            
            largest_patch_index = np.argmax([region.area for region in regions])
            
            # Extract coordinates of pixels in the largest patch
            largest_patch_coords = regions[largest_patch_index].coords
            
            # Fit an ellipse to the coordinates
            ellipse = cv2.fitEllipse(np.array(largest_patch_coords))
    
            # Calculate the ratio between major and minor axes
            major_axis, minor_axis = ellipse[1]
            aspect_ratio = major_axis / minor_axis
            
            dict_results["major_axis_ellipse"] = major_axis
            # Store the results in your dictionary
            dict_results["largest_patch_aspect_ratio"] = aspect_ratio
        # Count values for the current threshold
        count_over = np.sum(ds_patch["precipRateNearSurface"].data > threshold)
        dict_results[f"count_over_{threshold}"] = count_over
    
    for intensity_range in intensity_ranges:
        range_min, range_max = intensity_range

        # Mask pixels within the intensity range
        masked_pixels = np.logical_and(ds_patch["precipRateNearSurface"].data > range_min, ds_patch["precipRateNearSurface"].data <= range_max)

        # Count the number of pixels within the intensity range
        pixels_in_range = np.sum(masked_pixels)

        # Calculate the percentage
        percentage = (pixels_in_range / np.nansum(ds_patch["precipRateNearSurface"].data > 0)) * 100

        # Store the results in your dictionary
        dict_results[f"percentage_{range_min}_to_{range_max}"] = percentage


    
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
    dict_results["along_track_start"] = along_track_slice.start
    dict_results["along_track_end"] = along_track_slice.stop
    dict_results["gpm_granule_id"] = int(ds_patch["gpm_granule_id"][0].data)
    dict_results["time"] = ds_patch["time"][round(ds_patch["time"].data.shape[0] / 2)].data
    dict_results["sunLocalTime"] = float(ds_patch["sunLocalTime"][round(ds_patch["sunLocalTime"].data.shape[0] / 2)][round(ds_patch["sunLocalTime"].data.shape[1] / 2)].data)
    dict_results["lon"] = float(ds_patch["lon"][round(ds_patch["lon"].data.shape[0] / 2)][round(ds_patch["lon"].data.shape[1] / 2)].data)
    dict_results["lat"] = float(ds_patch["lat"][round(ds_patch["lat"].data.shape[0] / 2)][round(ds_patch["lat"].data.shape[1] / 2)].data)
    
    if along_track_slice.start == 0:
        dict_results["flag_granule_change"] = 1
    else: 
        dict_results["flag_granule_change"] = 0

    return dict_results
