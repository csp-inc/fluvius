from pystac_client import Client
import planetary_computer as pc
import os
import pandas as pd
import fsspec
import numpy as np
import geopandas as gpd
import argparse
from src.defaults import args_info

env_vars = open("/content/credentials","r").read().split('\n')

for var in env_vars[:-1]:
        key, value = var.split(' = ')
        os.environ[key] = value

storage_options={'account_name':os.environ['ACCOUNT_NAME'],\
                 'account_key':os.environ['BLOB_KEY']}
fs = fsspec.filesystem('az', account_name=storage_options['account_name'], account_key=storage_options['account_key'])

##env data acquired

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src',
        type=args_info["data_src"]["type"],
        help=args_info["data_src"]["help"])
    parser.add_argument('--write-to-csv',
        action=args_info["write_to_csv"]["action"],
        help=args_info["write_to_csv"]["help"])
    return parser

if __name__ == "__main__":

    args = return_parser().parse_args()

    if args.data_src == 'usgs':
        #USGS DATA PROCESS
        data_src = 'usgs'
        container = 'usgs-data'

        station_url = f'az://{container}/{args.data_src}_station_metadata_raw.csv'
        station_df = pd.read_csv(station_url, storage_options=storage_options)

        sites_str = [str(f).zfill(8) for f in station_df.site_no]
        station_df['sites_str'] = sites_str

        query = []
        for f in fs.ls(f'{container}/stations'):
            station = os.path.basename(f).split('_')[0]
            query.append(station)
        q = pd.DataFrame({'sites_str':query})
        out = station_df.merge(q, on='sites_str')
        out['site_no'] = out['sites_str']
        out = out[['site_no','site_name', 'Latitude', 'Longitude','geometry']]
        if args.write_to_csv:
            out.to_csv(f'az://{container}/usgs_station_metadata.csv',index=False, storage_options=storage_options)

    if args.data_src == 'ana':
        container = 'ana-data'
        station_url = f'az://{container}/ana_station_metadata.csv'
        station_df = pd.read_csv(station_url, storage_options=storage_options)
        for site_no in station_df.site_no:
            station_url = f'az://{container}/{site_no}.csv'
            station_url2 = f'az://{container}/{site_no}_2.csv'
            site_df1_raw = pd.read_csv(station_url, delimiter=',', skiprows=10, storage_options=storage_options)
            translation = pd.read_csv(f'az://{container}/ana_translations.csv', storage_options=storage_options)
            trans = {p:e  for p,e in zip(translation.Portuguese, translation.English)}
            site_df1 = site_df1_raw.rename(columns=trans)
            site_df1 = site_df1.dropna(subset=['Date'])
            site_df1['TimeL'] = site_df1['TimeL'].fillna('01/01/1900 01:00')
            site_df1['Date-Time'] = [d for d in site_df1['Date']]
            site_df1['Date-Time'] = pd.to_datetime(site_df1['Date-Time'],\
            format='%d/%m/%Y')

            site_df2_raw = pd.read_csv(station_url2, delimiter=',', skiprows=14, storage_options=storage_options)
            site_df2_raw = site_df2_raw.replace('01/01/1900', '01/01/1900 01:00')
            translation2 = {'Data':'Date','Hora':'Hour','Turbidez':'Turbidity'}
            site_df2 = site_df2_raw.rename(columns=translation2)
            site_df2 = site_df2.dropna(subset=['Date'])
            site_df2['Date-Time-HM'] = [f"{d} {t.split(' ')[1]}" for d,t in zip(site_df2['Date'],site_df2['Hour'])]
            site_df2['Date-Time'] = [d for d in site_df2['Date']]
            site_df2['Date-Time'] = pd.to_datetime(site_df2['Date-Time'],\
            format='%d/%m/%Y')
            site_df2 = site_df2[['Date', 'Hour', 'Date-Time','Turbidity']]

            selection = ['Date-Time', 'Discharge', 'Suspended Sediment Concentration (mg/L)', 'Turbidity']
            site_df = site_df1.merge(site_df2, on='Date', how='outer', suffixes=('_',''))
            site_df['Date-Time'] = site_df['Date-Time'].fillna(site_df['Date-Time_'])
            #site_df['Hour'] = site_df['Hour'].fillna(site_df['Hour_'])
            site_df = site_df[selection]
            s = str(site_no).zfill(8)
            write_filename = f'az://{container}/stations/{str(site_no)}.csv'
            print(f'writing to {write_filename}')
            if args.write_to_csv:
                site_df.to_csv(write_filename, index=False, storage_options=storage_options)
    
    if args.data_src == 'itv':
        container = 'itv-data'
        station_url = f'az://{container}/itv_station_metadata.csv'
        station_df = pd.read_csv(station_url, storage_options=storage_options)
        for site_no in station_df.site_no:
            station_url = f'az://{container}/{site_no}.csv'
            site_df = pd.read_csv(station_url,\
                                  storage_options=storage_options,\
                                  delimiter=',')

            site_df['Date-Time'] = pd.to_datetime(site_df['Campaign Date'], \
                                    format='%d/%m/%Y')

            if args.write_to_csv:
                write_filename = f'az://{container}/stations/{site_no}.csv'
                site_df.to_csv(write_filename, storage_options=storage_options,\
                                    index=False)
