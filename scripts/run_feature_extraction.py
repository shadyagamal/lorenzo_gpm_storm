#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:10:23 2023

@author: ghiggi
"""
import os
import dask
import logging
from dask.distributed import Client, LocalCluster
from gpm_storm.features.routines import run_granule_feature_extraction

dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})


@dask.delayed
def run_feature_extraction(filepath, dst_dir, force):
    with dask.config.set(scheduler="single-threaded"):
        try: 
            run_granule_feature_extraction(filepath, dst_dir=dst_dir, force=force)
        except Exception as e: 
            print(f"Processing of {filepath} failed with '{e}'.")
    return None 


if __name__ == "__main__": #  https://github.com/dask/distributed/issues/2520

    from gpm_api.io.local import get_local_filepaths

    
    dst_dir ="/ltenas8/data/GPM_STORM/features_v1"
    force = True 
    
    # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF/netCDF
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # Specify cluster settings
    cluster = LocalCluster(
        n_workers=24,
        threads_per_worker=1,
        processes=True,
        memory_limit="800GB",
        silence_logs=logging.WARN,
        )
    client = Client(cluster)

    # Retrieve GPM granule filepaths 
    filepaths = get_local_filepaths(product="2A-DPR", version=7, product_type="RS")
    
    filepaths = filepaths[0:4]
    # Apply storm patch feature extraction to each granule 
    list_delayed = [run_feature_extraction(filepath, dst_dir=dst_dir, force=force) for filepath in filepaths]
    
    # Compute delayed functions 
    list_delayed = dask.compute(list_delayed)

