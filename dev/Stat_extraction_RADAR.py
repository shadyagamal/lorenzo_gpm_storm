# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gpm_api
import datetime

# You can check that the config file has been correctly created with:
configs = gpm_api.read_configs()
print(configs)

gpm_api.available_products(product_category="RADAR", product_level="1B")

gpm_api.available_products(product_category="RADAR", product_level="2A")

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


variable1 = "precipRateNearSurface"
variable2 = "zFactorFinal"
da = ds[variable1]
da1 = ds[variable2]
da

# %%
print("Data type of numerical array: ", type(da.data.compute()))
da.data.compute()

ds["zFactorFinal"].sel(radar_frequency="Ka")


print("Is GPM ORBIT data?: ", ds.gpm_api.is_orbit)
print("Is GPM GRID data?: ", ds.gpm_api.is_grid)

list_slices = ds.gpm_api.get_slices_contiguous_scans()
print(list_slices)

slc = list_slices[0]
print(slc)

ds_regular = ds.isel(along_track=slc)

ds.gpm_api.is_spatial_2d  # because the xr.Dataset also contains the range and frequency dimensions !

ds["zFactorFinal"].isel(range=0).sel(radar_frequency="Ka").gpm_api.is_spatial_2d

ds["precipRateNearSurface"].gpm_api.is_spatial_2d

# %%

da.gpm_api.plot_map()


da.isel(along_track=slice(100, 300)).gpm_api.plot_map()


# %%
import numpy as np
import ximage  # noqa
import matplotlib.pyplot as plt

import gpm_api


# Available variables
variables = list(ds.data_vars)
# print(variables)


dir(ds.gpm_api)
ds.gpm_api.spatial_2d_variables

####---------------------------------------------------------------------------.
###################
#### Labelling ####
###################
min_value_threshold = 0.5
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

# Plot full label array
xr_obj[label_name].plot.imshow()  # 0 are plotted
plt.show()

# Plot label with ximage
xr_obj[label_name].ximage.plot_labels()

# Plot label with gpm_api
gpm_api.plot_labels(xr_obj[label_name])


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

# Define the patch generator
patch_gen = xr_obj.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    variable=variable1,
    # Output options
    n_patches=n_patches,
    n_labels=n_labels,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    padding=padding,
    centered_on=centered_on,
    # Tiling/Sliding Options
    partitioning_method=None,
)

# Plot patches around the labels
# list_da = list(patch_gen)
# label_id, da = list_da[0]
for label_id, da in patch_gen:
    p = gpm_api.plot_labels(
        da[label_name],
        add_colorbar=True,
        interpolation="nearest",
        cmap="Paired",
    )
    plt.show()

patch_gen = xr_obj.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    variable=variable1,
    # Output options
    n_patches=n_patches,
    n_labels=n_labels,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    padding=padding,
    centered_on=centered_on,
    # Tiling/Sliding Options
    partitioning_method=None,
)
list_label_patch = list(patch_gen)
label_id, da = list_label_patch[0]
# ----------------------------------------------------------------------------.
# %%
##################################
#### Label Patches Extraction ####
##################################
patch_size = (49, 20)

# Output Options
n_patches = 50
n_labels = None
labels_id = None
highlight_label_id = False
# Patch Extraction Options
centered_on = "label_bbox"
padding = 5
n_patches_per_label = np.Inf
n_patches_per_partition = 1
# Tiling/Sliding Options
partitioning_method = None
n_partitions_per_label = None
kernel_size = None
buffer = 0
stride = True
include_last = True
ensure_slice_size = False
debug = True
verbose = True

da_patch_gen = xr_obj.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    variable=variable1,
    # Output Options
    n_patches=n_patches,
    n_labels=n_labels,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch Extraction Options
    centered_on=centered_on,
    padding=padding,
    n_patches_per_label=n_patches_per_label,
    n_patches_per_partition=n_patches_per_partition,
    # Tiling/Sliding Options
    partitioning_method=partitioning_method,
    n_partitions_per_label=n_partitions_per_label,
    kernel_size=kernel_size,
    buffer=buffer,
    stride=stride,
    include_last=include_last,
    ensure_slice_size=ensure_slice_size,
    debug=debug,
    verbose=verbose,
)

gpm_api.plot_patches(da_patch_gen, variable=variable1, interpolation="nearest")


# list_patch = list(da_patch_gen)
# label_id, da_patch = list_patch[7]
# for label_id, da_patch in list_patch:
#      da_patch.gpm_api.plot_image()


# %%
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
    variable=variable1,
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

label_isel_dict

isel_dict = label_isel_dict[8][0]
da_patch = xr_obj.isel(isel_dict)


da_patch.gpm_api.plot_image()
da_patch.gpm_api.plot_map()


da_patch1 = da_patch.data[(da_patch["label"].data == 7)]


# %% patch statistics extraction
import numpy as np
import pandas as pd
import cv2


# Define a function to calculate statistics for a patch
def calculate_statistics(ds_patch):
    # Create a dictionary to store the results
    results = {}

    # extract data from ds_patch
    ds_patch["zFactorFinal"] = ds_patch["zFactorFinal"].compute()
    ds_patch["precipRateNearSurface"] = ds_patch["precipRateNearSurface"].compute()

    # Store the constants
    results["mean"] = np.mean(ds_patch["precipRateNearSurface"]).data
    results["std"] = np.std(ds_patch["precipRateNearSurface"]).data
    results["total_values"] = np.sum(ds_patch["precipRateNearSurface"].data > 0)
    results["sum_total_rain"] = ((ds_patch["precipRateNearSurface"]).data).sum()
    results["max_value"] = np.max(ds_patch["precipRateNearSurface"]).data

    # Threshold values to iterate over, including 0 for count_over_0
    thresholds = [0, 5, 10, 20, 50]

    # Iterate over thresholds and count_over variables
    for threshold in thresholds:
        # Count patches for the current threshold
        count_patches, _ = cv2.connectedComponents(
            (ds_patch["precipRateNearSurface"].data > threshold).astype(np.uint8), connectivity=8
        )
        results[f"count_patches_over_{threshold}"] = count_patches

        # Count values for the current threshold
        count_over = np.sum(ds_patch["precipRateNearSurface"].data > threshold)
        results[f"count_over_{threshold}"] = count_over
    # statistics on reflectivity
    ds_patch["REFCH"] = ds_patch.gpm_api.retrieve("REFCH").compute()
    ds_patch["echodepth18"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=18).compute()
    ds_patch["echodepth30"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=30).compute()
    ds_patch["echodepth50"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=50).compute()
    ds_patch["echotopheight18"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=18).compute()
    ds_patch["echotopheight30"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=30).compute()
    ds_patch["echotopheight50"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=50).compute()
    # ds_patch["SHI"] = ds_patch.gpm_api.retrieve("SHI")
    # ds_patch["MESH"] = ds_patch.gpm_api.retrieve("MESH")

    # Define a function to calculate statistics from data arrays with NaN values excluded
    def calculate_statistics(data):
        data = data[~np.isnan(data)]

        if data.size == 0:
            return np.nan, np.nan, np.nan

        mean = np.mean(data)
        std = np.std(data)
        max_value = np.max(data)

        return mean, std, max_value

    # Define a list of variable names and their corresponding data arrays
    variables = [
        ("REFCH", ds_patch["REFCH"].data),
        ("echodepth18", ds_patch["echodepth18"].data),
        ("echodepth30", ds_patch["echodepth30"].data),
        ("echodepth50", ds_patch["echodepth50"].data),
        ("echotopheight18", ds_patch["echotopheight18"].data),
        ("echotopheight30", ds_patch["echotopheight30"].data),
        ("echotopheight50", ds_patch["echotopheight50"].data),
    ]

    # Calculate statistics for each variable with NaN values excluded and store in the dictionary
    for var_name, data_array in variables:
        mean, std, max_value = calculate_statistics(data_array)

        # Store mean, std, and max in the results dictionary with appropriate keys
        results[f"{var_name}_mean"] = mean
        results[f"{var_name}_std"] = std
        results[f"{var_name}_max"] = max_value

    results["time"] = ds_patch["time"][round(ds_patch["time"].data.shape[0] / 2)].data

    results["lon"] = ds_patch["lon"][round(ds_patch["lon"].data.shape[0] / 2)][
        round(ds_patch["lon"].data.shape[1] / 2)
    ].data

    results["lat"] = ds_patch["lat"][round(ds_patch["lat"].data.shape[0] / 2)][
        round(ds_patch["lat"].data.shape[1] / 2)
    ].data

    # cut_count = ds_patch[]

    # Calculate statistics for each variable
    # refch_mean = ds_patch["REFCH"].data.mean(skipna=True)  # Calculate mean of REFCH
    # refch_std = ds_patch["REFCH"].data.std(skipna=True)  # Calculate standard deviation of REFCH
    # refch_max = ds_patch["REFCH"].data.max(skipna=True)  # Calculate maximum value of REFCH

    # echodepth_mean_18 = ds_patch["echodepth18"].data.mean(skipna=True)  # Calculate mean of echodepth
    # echodepth_std_18 = ds_patch["echodepth18"].data.std(skipna=True)  # Calculate standard deviation of echodepth
    # echodepth_max_18 = ds_patch["echodepth18"].data.max(skipna=True)  # Calculate maximum value of echodepth

    # echodepth_mean_30 = ds_patch["echodepth30"].data.mean(skipna=True)  # Calculate mean of echodepth
    # echodepth_std_30 = ds_patch["echodepth30"].data.std(skipna=True)  # Calculate standard deviation of echodepth
    # echodepth_max_30 = ds_patch["echodepth30"].data.max(skipna=True)  # Calculate maximum value of echodepth

    # echodepth_mean_50 = ds_patch["echodepth50"].data.mean(skipna=True)  # Calculate mean of echodepth
    # echodepth_std_50 = ds_patch["echodepth50"].data.std(skipna=True)  # Calculate standard deviation of echodepth
    # echodepth_max_50 = ds_patch["echodepth50"].data.max(skipna=True)  # Calculate maximum value of echodepth

    # echotopheight_mean_18 = ds_patch["echotopheight18"].data.mean(skipna=True)  # Calculate mean of echotopheight
    # echotopheight_std_18 = ds_patch["echotopheight18"].data.std(skipna=True)  # Calculate standard deviation of echotopheight
    # echotopheight_max_18 = ds_patch["echotopheight18"].data.max(skipna=True)  # Calculate maximum value of echotopheight

    # echotopheight_mean_30 = ds_patch["echotopheight30"].data.mean(skipna=True)  # Calculate mean of echotopheight
    # echotopheight_std_30 = ds_patch["echotopheight30"].data.std(skipna=True)  # Calculate standard deviation of echotopheight
    # echotopheight_max_30 = ds_patch["echotopheight30"].data.max(skipna=True)  # Calculate maximum value of echotopheight

    # echotopheight_mean_50 = ds_patch["echotopheight50"].data.mean(skipna=True)  # Calculate mean of echotopheight
    # echotopheight_std_50 = ds_patch["echotopheight50"].data.std(skipna=True)  # Calculate standard deviation of echotopheight
    # echotopheight_max_50 = ds_patch["echotopheight50"].data.max(skipna=True)  # Calculate maximum value of echotopheight

    # shi_mean = ds_patch["SHI"].mean().data.compute()  # Calculate mean of SHI
    # shi_max = ds_patch["SHI"].max().data.compute()  # Calculate maximum value of SHI

    # mesh_mean = ds_patch["MESH"].mean().data.compute()  # Calculate mean of MESH
    # mesh_max = ds_patch["MESH"].max().data.compute()  # Calculat

    return results


# Create an empty list to store statistics for each patch
patch_statistics = []

for i in range(1, n_patches):
    isel_dict = label_isel_dict[i][0]
    # reflectivity_patch = (ds["zFactorFinal"].sel(radar_frequency='Ku')).isel(isel_dict)
    ds_patch = ds.isel(isel_dict)

    # Calculate statistics for the patch
    stats = calculate_statistics(ds_patch)

    # Append the statistics to the list
    patch_statistics.append(stats)

# Create a pandas DataFrame from the list
df = pd.DataFrame(patch_statistics)


# %% correlation matrix
import seaborn as sns


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a larger heatmap of the correlation matrix

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")

# Display the plot
plt.title("Correlation Matrix")
plt.show()


# %% PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Data Preprocessing
# Assuming your data is in a 2D array or DataFrame with rows as samples and columns as variables.
df_cleaned = df.dropna(axis=1)  # Remove rows with missing values

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned)  # 'your_data' should be replaced with your data

# Step 2: PCA Calculation
n_components = 7  # Choose the number of components you want to retain
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_data)

# Step 3: Variance Explained
explained_variance_ratio = pca.explained_variance_ratio_
# You can assess how much variance is explained by each PC and decide on the number to retain.

# Step 4: Interpret and Use the PCs
# You can access the loadings using 'pca.components_' and use the transformed data in 'principal_components' for further analysis or visualization.

# Create a scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)

# Add labels to some data points for illustration (you can customize this)
for i, (x, y) in enumerate(zip(principal_components[:, 0], principal_components[:, 1])):
    if i == 0:
        plt.text(x, y, "A", fontsize=12, ha="right", va="bottom")
    elif i == 1:
        plt.text(x, y, "B", fontsize=12, ha="left", va="top")
    # Add more annotations as needed

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - First Two Principal Components")

# Annotate clusters or outliers if identified
# You can add code to identify and annotate clusters or outliers based on your data.

plt.grid()
plt.show()


# Variance explained plot
explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio_cumsum,
    marker="o",
    linestyle="--",
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained by Principal Components")
plt.grid()
plt.show()


# %% k-means

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler
scaler = StandardScaler()

# Fit and transform the data
df_scaled = scaler.fit_transform(df_cleaned)


# Define the number of clusters (k)
k = 3

# Create a K-Means model
kmeans = KMeans(n_clusters=k)

# Fit the model to the data
kmeans.fit(df_scaled)

# Get the cluster assignments for each data point
labels = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Visualize the clustering results
plt.figure(figsize=(8, 6))

# Scatter plot of data points with different colors for each cluster
for i in range(k):
    plt.scatter(
        df_scaled[labels == i, 0], df_scaled[labels == i, 1], label=f"Cluster {i + 1}", alpha=0.5
    )

# Plot cluster centers
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1], c="black", marker="x", s=40, label="Centroids"
)

plt.title("K-Means Clustering")
plt.legend()
plt.grid()
plt.show()


# You can access the cluster assignments for each data point
labels = kmeans.labels_
