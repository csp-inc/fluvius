import sys, os
import pandas as pd
from rasterio.enums import Resampling

# Append content dir to path to support import of scripts
sys.path.append("/content")
from src.fluvius import *

# Gather and parce environment variables to create storage options for
# blob storage access via pandas
with open("/content/.env") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(' = ')
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"],
                   "connection_string": os.environ["CONNECTION_STRING"]}

# Pull USGS water data
usgswd = WaterData("usgs", "usgs-data", storage_options)
usgswd.get_source_df()
usgswd.apply_buffer_to_points(100)
usgswd.get_station_data()

# Now just grab a single station to demonstrate the WaterStation methods
my_station = usgswd.station["01632900"]

# Build the pystac catalog of images matching observations (space and time)
my_station.build_catalog()

# collect info for tiles meeting cloud criteria and spatio-temporal overlap 
# with station sites and merge hrefs with main df
my_station.get_cloud_filtered_image_df(0.85)
my_station.merge_image_df_with_samples()

# Extract reflectances and append to my_station.merged_df as new columns
my_station.get_chip_features()

# Check out the merged dataframe
my_station.merged_df
