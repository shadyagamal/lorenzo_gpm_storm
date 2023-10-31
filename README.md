# Welcome to GPM-STORM


## Quick start


Before starting using GPM-API, we highly suggest to save into a configuration file:
1. your credentials to access the [NASA Precipitation Processing System (PPS) servers][PPS_link]
2. the directory on the local disk where to save the GPM dataset of interest.

To facilitate the creation of the configuration file, you can run the following script:

```python
import gpm_api

username = "<your PPS username>" # likely your mail
password = "<your PPS password>" # likely your mail
gpm_base_dir = "<path/to/directory/GPM"  # path to the directory where to download the data
gpm_api.define_configs(username_pps=username,
                       password_pps=password,
                       gpm_base_dir=gpm_base_dir)

# You can check that the config file has been correctly created with:
configs = gpm_api.read_configs()
print(configs)

```
