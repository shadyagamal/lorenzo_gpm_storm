#!/usr/bin/env python3
"""
Created on Tue Oct 31 15:06:41 2023

@author: ghiggi
"""
import gpm_api

username = "lorenzo.comi@epfl.ch"  # likely your mail
password = "lorenzo.comi@epfl.ch"  # likely your mail
gpm_base_dir = "/home/comi/data/GPM"  # path to the directory where to download the data
gpm_api.define_configs(gpm_username=username, gpm_password=password, gpm_base_dir=gpm_base_dir)
