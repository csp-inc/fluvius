import sys, os
import pandas as pd
sys.path.append("/fluvius")
from src.fluvius import *

## Import env vars to get storage options for accessing Azure blob with pandas
with open("/fluvius/.env") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(' = ')
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"],
                   "connection_string": os.environ["CONNECTION_STRING"]}

## Start looking at data
usgswd = WaterData("usgs", "usgs-data", storage_options)
usgswd.get_source_df()
usgswd.apply_buffer_to_points(100)
usgswd.get_station_data()