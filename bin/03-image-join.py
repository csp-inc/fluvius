### Environment setup
import sys, os
sys.path.append('/content')
from src.fluvius import WaterData
import fsspec
import pandas as pd
import argparse
from concurrent import futures
import matplotlib as plt

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
env_vars = open("/content/credentials","r").read().split('\n')

for var in env_vars[:-1]:
    key, value = var.split(' = ')
    os.environ[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src',\
        type=str,\
        help="name of data source")
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
        help="search radius to use for reflectance data aggregation")
    parser.add_argument('--write_to_csv',\
        default=False,\
        type=bool,\
        help="Write out csvs to ./data?")
    parser.add_argument('--write_chips',\
        default=False,\
        type=bool,\
        help="Write chips to blob storage?")
    parser.add_argument('--mask_method',\
        default="lulc",\
        type=str,\
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    args = parser.parse_args()

    #################  set up ####################
    data_source = args.data_src
    container = f'{data_source}-data'

    ############# initial parameters #############
    if data_source == 'usgs':
        day_tolerance = 0 #reduce this for usgs-data
    else:
        day_tolerance = args.day_tolerance 
    
    cloud_thr = args.cloud_thr
    buffer_distance = args.buffer_distance
    blob_dir = f"chips/{buffer_distance}m_cloudthr{cloud_thr}_{args.mask_method}_masking"
    ################### Begin ####################

    storage_options={'account_name':os.environ['ACCOUNT_NAME'],\
                     'account_key':os.environ['BLOB_KEY']}

    fs = fsspec.filesystem('az',\
                            account_name=storage_options['account_name'],\
                            account_key=storage_options['account_key'])  
    ds = WaterData(data_source, container, storage_options)
    ds.get_source_df()
    ds.apply_buffer_to_points(buffer_distance)
    
    # Define a function for getting station feature data in parallel
    def get_station_feature_df(station, cloud_thr, day_tol):
        ds.get_station_data(station)
        ds.station[station].drop_bad_usgs_obs()
        ds.station[station].build_catalog()
        if ds.station[station].catalog is None:
            print(f"No matching images for station {station}. Skipping...")
            return
        else:
            ds.station[station].get_cloud_filtered_image_df(cloud_thr)
            ds.station[station].merge_image_df_with_samples(day_tol)
            ds.station[station].perform_chip_cloud_analysis()
            ds.station[station].get_chip_features(args.write_chips, blob_dir, args.mask_method)
        if args.write_to_csv:
            sstation = str(station).zfill(8)
            outfilename = f'az://{ds.container}/stations/{sstation}/{sstation}_processed_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent.csv'
            ds.station[station].merged_df.to_csv(
                outfilename,index=False,
                storage_options=ds.storage_options)
            print(f'wrote csv to {outfilename}')
            # print('writing chips!')
            # ds.station[station].write_tiles_to_blob(working_dirc='/tmp')

    stations = ds.df["site_no"]
    cloud_threshold = cloud_thr
    day_tol = day_tolerance
    for station in stations:
        get_station_feature_df(station, cloud_threshold, day_tol)
    
    ## Merge dataframes w/ feature data for all stations, write to blob storage
    print("Merging station feature dataframes and saving to blob storage.")
    df = pd.DataFrame()
    for station in stations:
        if station in ds.station:
            try:
                df = pd.concat([df, ds.station[station].merged_df.reset_index()], axis=0)
            except:
                print(f"no attribute merged_df for station {station}")
        else:
            continue

    outfileprefix = f"az://modeling-data/{ds.container}/feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{args.mask_method}_masking"
    
    df.to_csv(f"{outfileprefix}.csv", storage_options=storage_options)
    print("Done!")