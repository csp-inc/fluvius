### Environment setup
import sys, os
sys.path.append('/content')
from src.fluvius import WaterData
import fsspec
import pandas as pd
import argparse
from concurrent import futures

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
        default=0.8,\
        type=float,\
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',\
        default=500,\
        type=int,\
        help="search radius to use for reflectance data aggregation")
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
            ds.station[station].get_chip_features()
        if args.write_to_csv:
            sstation = str(station).zfill(8)
            outfilename = f'az://{ds.container}/stations/{sstation}/{sstation}_processed_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{int(cloud_thr*100)}percent.csv'
            ds.station[station].merged_df.to_csv(
                outfilename,index=False,
                storage_options=ds.storage_options)
            print(f'wrote csv {outfilename}')
            # print('writing chips!')
            # ds.station[station].write_tiles_to_blob(working_dirc='/tmp')

    stations = ds.df["site_no"]
    cloud_threshold = [cloud_thr] * len(stations)
    day_tol = [day_tolerance] * len(stations)

    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as pool:
        pool.map(get_station_feature_df, stations, cloud_threshold, day_tol)
    
    ## Merge dataframes w/ feature data for all stations, write to blob storage
    print("Merging station feature dataframes and saving to blob storage.")
    df = pd.DataFrame()
    for station in stations[0:]:
        if ds.station[station].catalog is None:
            continue
        feature_df = pd.concat([df, ds.station[station].merged_df.reset_index()], axis=0)

    outfileprefix = f"az://{ds.container}/merged_station_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{int(cloud_thr*100)}percent"

    df.to_csv(f"{outfileprefix}.csv", storage_options=storage_options)
    df.to_json(f"{outfileprefix}.json", storage_options=storage_options)

    print("Done!")