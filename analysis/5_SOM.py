#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:03:56 2019

@author: ghiggi
"""

from pathlib import Path
from netCDF4 import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

import umap
import somoclu
#------------------------------------------------------------------------------.
figs_path = Path('/home/comi/Project_LTE/gpm_storm/gpm_storm/images')
Data_path = Path('/home/comi/Project_LTE/gpm_storm/gpm_storm/data')



# Initialize lists to store image data
image_data = []
image_filenames = []

# Iterate through the files in the directory
for filename in os.listdir(figs_path):
    if filename.endswith(".png"):
        # Open the image and convert it to a NumPy array
        image = Image.open(os.path.join(figs_path, filename))
        image_array = np.array(image)

        # Append the image data and filename to lists
        image_data.append(image_array)
        image_filenames.append(filename)

# Create a Pandas DataFrame
df_img = pd.DataFrame({'Filename': image_filenames, 'Image Data': image_data})



# img_dat = xr.open_dataset(Path(Data_path, 'masc_davos.nc'))
# img_dat.images.shape
 
# img = img_dat.images[1,:,:,0]
# plt.imshow(img, cmap="gray")

latent_data = pd.read_parquet(Path(Data_path, 'dpr_feature.parquet'), engine='pyarrow')
latent_data_array = latent_data.to_numpy(dtype=np.float32)
# latent_data = xr.open_dataset(Path(Data_path, 'masc_latent_dist.nc'))
# latent_codes = latent_data.latent_mean   # 8 dimensions 
# latent_cov = latent_data.latent_cov
#-----------------------------------------------------------------------------.
#################### 
## SOM training ####
####################
# Define SOM grid 
n_rows, n_columns = 4,4
# Define SOM structure 
# - maptype : "toroid","planar"
# - gridtype . "hexagonal", "rectangular"
som = somoclu.Somoclu(n_columns=n_columns, n_rows=n_rows, \
                      gridtype='rectangular', maptype='planar') #  initialcodebook)  ...sample the original codes 
#-----------------------------------------------------------------------------.
# Train SOM
# train(data=None, epochs=10,  scale0=0.1, scaleN=0.001, scalecooling='linear')
som.train(data=latent_data_array[:,:20], epochs=10, \
          radius0=0, radiusN=1, \
          scale0=0.5, scaleN=0.001)
#-----------------------------------------------------------------------------.
# Retrieve som codebooks (nodes centroids)
#codebooks = som.codebook 
#codebooks.shape
## Retrieve Umatrix 
#Umatrix = som.umatrix 
#Umatrix.shape
## Retrieve the activation map  
## - Euclidean distance between input and codebooks  
##ActivationMatrix = som.activation_map
## ActivationMatrix.shape  # (nobs x n_nodes)
## Retrieve position of input obs in the map  (BMUs)
#node_assignement = som.bmus 
#node_assignement.shape
## Get distance between data and codebooks 
#newdata = latent_codes
#ActivationMatrix_new = som.get_surface_state(data=newdata)
## Get Best Matching Units indexes of the activation map 
#new_data_assignement = som.get_bmus(ActivationMatrix_new)
##-----------------------------------------------------------------------------.
### Provide new data and update the som 
## som.update_data(data=newdata)
## som.train()
##-----------------------------------------------------------------------------.
## Display the node centroid/codebook values for each of the dimension
#som.view_component_planes(dimensions)
## Display activation map (of one obs) (aka distance to codebooks)
#som.view_activation_map(data_index=1) 
## Display U-matrix
#colors = ["red"] * 50
#colors.extend(["green"] * 50)
#colors.extend(["blue"] * 50)
#labels = range(150)
#som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)
## Display the similarity matrix 
## - som.view_similarity_matrix()
##-----------------------------------------------------------------------------.
#codebooks = som.codebook 
#codebooks.shape
#
#
################################
## Clustering the SOM nodes ####
################################
#from sklearn.cluster import DBSCAN
#algorithm = DBSCAN()
#som.cluster(algorithm=algorithm)
#
#plt.imshow(som.clusters)
#som.view_umatrix(bestmatches=True)
##-----------------------------------------------------------------------------.
####################### 
### Plot MASC grid ####
#######################
#images = img_dat.images[:,:,:,0]
#images.shape
## Retrieve node exemplar 
#node_assignement = som.bmus 
#node_assignement.shape
#nodes_selected, img_idx_sample = np.unique(node_assignement, return_index=True, axis=0)
### Plot snowflakes 
#def cm2inch(*tupl):
#    inch = 2.54
#    if isinstance(tupl[0], tuple):
#        return tuple(i/inch for i in tupl[0])
#    else:
#        return tuple(i/inch for i in tupl)
#    
#from matplotlib import gridspec
#gs = gridspec.GridSpec(20,20)
#
#plt.figure(figsize=cm2inch(20, 20), dpi=1000)
#for ((i,j),ind) in zip(nodes_selected,img_idx_sample):
#    plt.subplot(gs[i,j])
#    plt.imshow(images[ind,:,:], cmap="gray")
#    plt.gca().get_xaxis().set_visible(False)
#    plt.gca().get_yaxis().set_visible(False)
#plt.savefig(Path(figs_path, 'MASC_Organized'))
#plt.close()

# plt.gca().tick_params?
################################# 
## Plot MASC grid color axis ####
#################################

# Define custom distance
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))
    
# Retrieve distance with matrix neighbors 
def retrieve_nearest_neighbor_distance(codebooks, row, col, dist):
    dim_codebooks = codebooks.shape  # (nrow, ncol, dim_variables)
    # ----------------------------------------------------------------------.
    # Above (top)
    tmp_row = row - 1
    tmp_col = col
    if tmp_row < 0 or row == 0:
        above_dist = float('NaN') 
    else: 
        above_dist = dist(codebooks[row,col,:], codebooks[tmp_row, tmp_col,:])
    # ----------------------------------------------------------------------.
    # Below (bottom)
    tmp_row = row + 1
    tmp_col = col
    if tmp_row > (dim_codebooks[0] - 1) or row == (dim_codebooks[0]-1):
        below_dist = float('NaN') 
    else: 
        below_dist = dist(codebooks[row,col,:], codebooks[tmp_row, tmp_col,:])
    # ----------------------------------------------------------------------.
    # Right
    tmp_col = col + 1 
    tmp_row = row
    if tmp_col > (dim_codebooks[1] - 1) or col == (dim_codebooks[1]-1):
        right_dist = float('NaN') 
    else: 
        right_dist = dist(codebooks[row,col,:], codebooks[tmp_row, tmp_col,:])
    # ----------------------------------------------------------------------.
    # Left 
    tmp_col = col - 1
    tmp_row = row
    if tmp_col < 0 or col == 0:
        left_dist = float('NaN') 
    else: 
        left_dist = dist(codebooks[row,col,:], codebooks[tmp_row, tmp_col,:])
    # ----------------------------------------------------------------------.    
    d = dict([("top",above_dist), ("bottom",below_dist),("right", right_dist),("left",left_dist)])    
    return(d)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

## Define colorbar and min max
# - mpl.cm.<Spectral> : colormaps are defined by default between 0 and 255  mpl.cm.Spectral(257)
def get_colors_from_cmap(x, cmap_name='Spectral', vmin=None, vmax=None): 
 flag_dict = False
 # Preprocess x if dictionary
 if (isinstance(x, dict)):
     flag_dict = True
     keys = list(x.keys())
     x = np.asarray(list(x.values()))
 # Get index with NaN 
 idx_nan = np.isnan(x)  
 # Retrieve colormap 
 cmap = mpl.cm.get_cmap(cmap_name)
 # Rescale x and assign colormap values 
 rgb_val = cmap(((x - vmin)/vmax))
 # norm_fun = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
 # rgb_val = cmap(norm_fun(x)) 
 #----------------------------------------------------------------.
 ######################
 ## Convert to hex ####
 ######################
 # If only a single value
 if isinstance(rgb_val, tuple):
    rgb_hex = mpl.colors.rgb2hex(rgb_val)
 # If multiple values (matrix with rows of rgb values)
 else:
    rgb_hex = []
    for i in range(rgb_val.shape[0]):
        rgb_hex.append(mpl.colors.rgb2hex(rgb_val[i,]))
 #----------------------------------------------------------------.
 # Set back NaN
 rgb_hex = np.array(rgb_hex)
 if np.sum(idx_nan) > 0:
   rgb_hex[idx_nan] = 'NaN'
 #################################################
 # If x is dictionary --> Recreate dictionary ####
 #################################################
 if flag_dict:
    rgb_hex = dict(zip(keys, rgb_hex))
 #---------------------------------------------------------------------.
 return(rgb_hex)      
 
 #### Test get_colors_from_cmap and retrieve_nearest_neighbor_distance
#get_colors_from_cmap(2, cmap_name="Spectral", vmin=0, vmax = 8)
#get_colors_from_cmap(np.array([2,np.nan,5]), cmap_name="Green", vmin=0, vmax = 8)
#get_colors_from_cmap(np.array([0,2,3,5]), cmap_name="Wistia", vmin=0, vmax = 5)
#retrieve_nearest_neighbor_distance(codebooks,row=18, col=19, dist=dist) 
#------------------------------------------------------------------------------.
images = df_img['Image Data']
images = images.to_numpy()
images.shape
# Retrieve node exemplar 
node_assignement = som.bmus 
node_assignement.shape
nodes_selected, img_idx_sample = np.unique(node_assignement, return_index=True, axis=0)
codebooks = som.codebook

## Define minimum and max distance 
vmin = 0
vmax = 0.6


## Plot snowflakes 
from matplotlib import gridspec
gs = gridspec.GridSpec(25,25)
gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

plt.figure(figsize=cm2inch(25,25), dpi=1000)
for ((i,j),ind) in zip(nodes_selected,img_idx_sample):
    tmp_dist_dict = retrieve_nearest_neighbor_distance(codebooks,row=i, col=j, dist=dist)
    tmp_colors_dict = get_colors_from_cmap(tmp_dist_dict, cmap_name='Spectral', vmin=vmin,vmax=vmax)   
   
    # Plot  
    plt.subplot(gs[i,j])
    plt.imshow(images[ind], cmap="gray")
    # Set the color of the axis 
   # Set the color of the axis
    for position in ['bottom', 'left', 'top', 'right']:
        color = tmp_colors_dict.get(position, 'default_color')  # Replace 'default_color' with a valid color
        if color == 'NaN':
            color = 'red'  # Replace 'default_color' with a valid color
        plt.gca().spines[position].set_color(color)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    
        # Set consistent axis limits for all subplots
        # plt.xlim(x_min, x_max)  # Replace x_min and x_max with your desired x-axis limits
        # plt.ylim(y_min, y_max)  # Replace y_min and y_max with your desired y-axis limits

 
    
# Save the figure     
plt.savefig(Path(figs_path, 'MASC_Organized'))
 
 
 
         
         
