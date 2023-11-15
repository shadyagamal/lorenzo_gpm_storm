"""
Spyder Editor

This is a temporary script file.
"""
import datetime

import gpm_api
import numpy as np
import pandas as pd
import ximage  # noqa
#import gpm_storm
from matplotlib import pyplot as plt

from gpm_storm.features.image import calculate_image_statistics
from gpm_api.io.local import get_local_filepaths


# Specify the time period you are interested in
start_time = datetime.datetime.strptime("2020-07-22 05:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-07-22 12:00:00", "%Y-%m-%d %H:%M:%S")
# Specify the product and product type
product = "2A-DPR"  # 2A-PR
product_type = "RS"
# Specify the version
version = 7

# %%
# Download the data
gpm_api.download(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    force_download=False,
    verbose=True,
    progress_bar=True,
    check_integrity=False,
)


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


filepaths = get_local_filepaths(product, version=7, product_type="RS")


for filepath in filepaths:
    ds = gpm_api.open_granule(filepath, variables=variables, scan_mode="FS")


    # Load the dataset
    ds = gpm_api.open_dataset(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        variables=variables,
        prefix_group=False,
    )
    ds
    
    
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
    
    # Plot full label array
    # xr_obj[label_name].plot.imshow()  # 0 are plotted
    # plt.show()
    
    # # Plot label with ximage
    # xr_obj[label_name].ximage.plot_labels()
    
    # # Plot label with gpm_api
    # gpm_api.plot_labels(xr_obj[label_name])
    
    
    ####---------------------------------------------------------------------------.
    ##############################
    #### Visualize each label ####
    ##############################
    patch_size = (49, 20)
    # Output Options
    n_patches = 50
    label_name = "label"
    highlight_label_id = False
    labels_id = None
    n_labels = None
    # Patch Extraction Options
    centered_on = "label_bbox"
    padding = 0
    variable = "precipRateNearSurface"
    
    # #Define the patch generator
    # patch_gen = xr_obj.ximage.label_patches(
    #     label_name=label_name,
    #     patch_size=patch_size,
    #     variable=variable,
    #     # Output options
    #     n_patches=n_patches,
    #     n_labels=n_labels,
    #     labels_id=labels_id,
    #     highlight_label_id=highlight_label_id,
    #     # Patch extraction Options
    #     padding=padding,
    #     centered_on=centered_on,
    #     # Tiling/Sliding Options
    #     partitioning_method=None,
    # )
    # #Plot patches around the labels
    # list_da = list(patch_gen)
    # label_id, da = list_da[0]
    # for label_id, da in patch_gen:
    #     p = gpm_api.plot_labels(
    #         da[label_name],
    #         add_colorbar=True,
    #         interpolation="nearest",
    #         cmap="Paired",
    #     )
    #     plt.show()
    
    # ----------------------------------------------------------------------------.
    # %%
    ##################################
    #### Label Patches Extraction ####
    ##################################
    patch_size = (49, 20)
    
    # Output Options
    n_patches = 50
    label_name = "label"
    highlight_label_id = False
    labels_id = None
    n_labels = None
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
    
    # Save DataFrame to Parquet
    filepath = f"/home/comi/Project_LTE/gpm_storm/gpm_storm/dpr_feature_granule_{filepath}.parquet"  # f"feature_{granule_id}.parquet"
    df.to_parquet(filepath)
