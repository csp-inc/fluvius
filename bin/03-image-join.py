### Environment setup
import sys
sys.path.append('/content')
from src.fluvius import WaterData, WaterStation

import pandas as pd
import numpy as np
import geopandas as gpd
import fsspec
from pystac_client import Client
import planetary_computer as pc
import os

import matplotlib.pyplot as plt
# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)

env_vars = open("/content/.env","r").read().split('\n')

for var in env_vars[:-1]:
    key, value = var.split(' = ')
    os.environ[key] = value

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        parser.add_argument('--data_src',\
        type=str,\
        help="name of data type")
    parser.add_argument('--day_tolerance',\
        default=8,\
        type=int,\
        help="days of search around sample date")
    parser.add_argument('--cloud_thr',\
        default=80,\
        type=int,\
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',\
        default=500,\
        type=int,\
        help="meters search distance around point of interst")
    parser.add_argument('--write-to-csv',\
        default=False,\
        type=bool,\
        help="Write out csvs to ./data")
    args = parser.parse_args()


    #################  set up ####################
    data_source = args.data_src
    container = f'{data_source}-data'
    ############## initial parameters ##############
    if data_source == 'usgs':
        day_tolerance = 0 #reduce this for usgs-data
    else:
        day_tolerance = args.day_tolerance 
        cloud_thr = args.cloud_thr
        buffer_distance = args.buffer_distance # change this to increase chip size in meter 
    ################################################

    storage_options={'account_name':os.environ['ACCOUNT_NAME'],\
                    'account_key':os.environ['BLOB_KEY'],
                    'connection_string': os.environ['CONNECTION_STRING']}

    fs = fsspec.filesystem('az',\
                            account_name=storage_options['account_name'],\
                            account_key=storage_options['account_key'])  
    WaterData(data_source, container, storage_options)
    ds.get_source_df()
    ds.apply_buffer_to_points(buffer_distance)
    #full loop
    for station in ds.df['site_no']:
        ds.get_station_data(station)
        ds.station[station].build_catalog()
        if ds.station[station].catalog is None:
            print('No matching images! Skipping...')
            continue
        else:
            ds.station[station].get_cloud_filtered_image_df(cloud_thr)
            ds.station[station].merge_image_df_with_samples(day_tolerance)
            ds.station[station].perform_chip_cloud_analysis()
            ds.station[station].get_reflectances()
        if args.write_to_csv:
            sstation = str(station).zfill(8)
            outfilename = f'az://{ds.container}/stations/{sstation}/{sstation}_processed.csv'
            ds.station[station].merged_df.to_csv(outfilename,index=False,storage_options=ds.storage_options)
            print(f'wrote csv {outfilename}')
            print('writing chips!')
            ds.station[station].write_tiles_to_blob(working_dirc='/tmp')
