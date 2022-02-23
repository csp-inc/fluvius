### Environment setup
import sys, os
sys.path.append('/content')
from src.fluvius import WaterData
import fsspec
import pandas as pd
import argparse
import shutil
from src.defaults import args_info

import faulthandler
faulthandler.enable()

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
env_vars = open("/content/credentials","r").read().split('\n')

for var in env_vars[:-1]:
    key, value = var.split(' = ')
    os.environ[key] = value

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src',
        type=args_info["data_src"]["type"],
        choices=args_info["data_src"]["choices"],
        help=args_info["data_src"]["help"])
    parser.add_argument('--day-tolerance',
        default=args_info["day_tolerance"]["default"],
        type=args_info["day_tolerance"]["type"],
        help=args_info["day_tolerance"]["help"])
    parser.add_argument('--cloud-thr',
        default=args_info["cloud_thr"]["default"],
        type=args_info["cloud_thr"]["type"],
        help=args_info["cloud_thr"]["help"])
    parser.add_argument('--buffer-distance',
        default=args_info["buffer_distance"]["default"],
        type=args_info["buffer_distance"]["type"],
        help=args_info["buffer_distance"]["help"])
    parser.add_argument('--write-to-csv',
        action=args_info["write_to_csv"]["action"],
        help=args_info["write_to_csv"]["help"])
    parser.add_argument('--write-chips',
        action=args_info["write_chips"]["action"],
        help=args_info["write_chips"]["help"])
    parser.add_argument('--mask-method1',
        default=args_info["mask_method1"]["default"],
        type=args_info["mask_method1"]["type"],
        choices=args_info["mask_method1"]["choices"],
        help=args_info["mask_method1"]["help"])
    parser.add_argument('--mask-method2',
        default=args_info["mask_method2"]["default"],
        type=args_info["mask_method2"]["type"],
        choices=args_info["mask_method2"]["choices"],
        help=args_info["mask_method2"]["help"])
    return parser

if __name__ == "__main__":
    
    args = return_parser().parse_args()

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
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    local_save_dir = f"data/chips/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking"

    ################### Begin ####################
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    else: # remove for data source if exists to start fresh
        if os.path.exists(f"{local_save_dir}/{data_source}"):
            shutil.rmtree(f"{local_save_dir}/{data_source}") 
        
    
    storage_options= {'account_name':os.environ['ACCOUNT_NAME'],
                      'account_key':os.environ['BLOB_KEY']}

    fs = fsspec.filesystem('az',
                            account_name=os.environ['ACCOUNT_NAME'],
                            account_key=os.environ['BLOB_KEY'])
    ds = WaterData(data_source, container, buffer_distance, storage_options)
    ds.get_source_df()

    # Getting station feature data in for loop
    stations = ds.df["site_no"]
    cloud_threshold = cloud_thr
    day_tol = day_tolerance
    for station in stations:
        try:
            ds.get_station_data(station)
            ds.station[station].drop_bad_usgs_obs()
            ds.station[station].build_catalog()
            if ds.station[station].catalog is None:
                print(f"No matching images for station {station}. Skipping...")
                continue
            else:
                ds.station[station].get_cloud_filtered_image_df(cloud_thr)
                ds.station[station].merge_image_df_with_samples(day_tol)
                if len(ds.station[station].merged_df) == 0:
                    print(f"No cloud-free images for station {station}. Skipping...")
                    continue

                ds.station[station].perform_chip_cloud_analysis()
                ds.station[station].get_chip_features(args.write_chips, local_save_dir, mm1, mm2)
            if args.write_to_csv:
                sstation = str(station).zfill(8)
                outfilename = f'az://{ds.container}/stations/{sstation}/{sstation}_processed_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent.csv'
                ds.station[station].merged_df.to_csv(
                    outfilename,index=False,
                    storage_options=ds.storage_options)
                print(f'wrote csv to {outfilename}')
        except FileNotFoundError:
            print(f"Source file not found for station {station}. Skipping...")
    
    ### Upload the chips to blob storage
    print("Uploading chips to blob storage.")
    blob_dir = f"modeling-data/chips/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_source}/"
    try:
        fs.rm(blob_dir, recursive=True)
    except:
        pass
    
    fs.put(f"{local_save_dir}/{data_source}/", blob_dir, recursive=True)

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

    outfileprefix = f"az://modeling-data/{ds.container}/feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking"

    df.to_csv(f"{outfileprefix}.csv", storage_options=storage_options)
    print("Done!")
