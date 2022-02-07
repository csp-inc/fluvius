### Environment setup
import sys, os
sys.path.append('/content')
from src.fluvius import WaterData
import fsspec
import pandas as pd
import argparse
import shutil
import datetime
import faulthandler
faulthandler.enable()

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
env_vars = open("/content/credentials","r").read().split('\n')

for var in env_vars[:-1]:
    key, value = var.split(' = ')
    os.environ[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src',
        type=str,
        choices=["itv", "ana", "usgs", "usgsi"],
        help="name of data source")
    parser.add_argument('--cloud_thr',
        default=80,
        type=int,
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',
        default=500,
        type=int,
        help="search radius to use for reflectance data aggregation")
    parser.add_argument('--write_chips',
        default=False,
        type=bool,
        help="Write chips to blob storage?")
    parser.add_argument('--mask_method1',
        default="lulc",
        choices=["lulc", "scl"],
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',
        default="",
        choices=["ndvi", "mndwi", ""],
        type=str,
        help="Which additional index, if any, to use to update the mask, (\"ndvi\") or (\"mndwi\")")
    parser.add_argument('--start_date',
        default="2015-01-01",
        type=str,
        help="The earliest date for which to generate prediction inputs")
    parser.add_argument('--end_date',
        default="2021-12-31",
        type=str,
        help="The latest date for which to generate prediction inputs")
    args = parser.parse_args()

    #################  set up ####################
    data_source = args.data_src
    container = f'{data_source}-data'

    ############# initial parameters #############
    day_tolerance = 0

    cloud_thr = args.cloud_thr
    buffer_distance = args.buffer_distance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    start_date = datetime.date.fromisoformat(args.start_date)
    end_date = datetime.date.fromisoformat(args.end_date)

    local_save_dir = f"data/prediction-chips/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/"

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
        if (station in ["ITV1", "ITV2"]) and (mm1 == "lulc"): # lulc has no water pixels for these sites
            continue

        try:
            ds.get_station_data(station)
            daterange = pd.date_range(
                start_date,
                end_date - datetime.timedelta(days=1),
                freq='d').to_pydatetime()

            dates = [x.date() for x in daterange]

            lat = [ds.station[station].df["Latitude"].iloc[0]] * len(dates)
            lon = [ds.station[station].df["Longitude"].iloc[0]] * len(dates)

            sample_ids = [f"{ds.station[station].site_no.zfill(8)}_pred_{str(i).zfill(8)}" for i in range(0, len(dates))]

            ds.station[station].df = pd.DataFrame({
                "sample_id": sample_ids,
                "Longitude": lon,
                "Latitude": lat,
                "Date-Time": dates
            })

            ds.station[station].time_of_interest = f"{args.start_date}/{args.end_date}"
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
        except FileNotFoundError:
            print(f"Source file not found for station {station}. Skipping...")

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

    outfileprefix = f"az://prediction-data/{ds.container}/feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking"

    df.to_csv(f"{outfileprefix}.csv", storage_options=storage_options)

    ## Upload the chips to blob storage
    print("Uploading chips to blob storage.")
    if args.write_chips:
        blob_dir = f"prediction-data/chips/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_source}/"
        try:
            fs.rm(blob_dir, recursive=True)
        except:
            pass
        
        fs.put(f"{local_save_dir}/{data_source}/", blob_dir, recursive=True)

    print("Done!")
