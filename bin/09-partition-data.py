import pandas as pd
import numpy as np
import os, sys
sys.path.append("/content")
from src.utils import train_test_validate_split
import datetime as dt
import argparse

if __name__ == "__main__":
    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--day_tolerance',\
        default=8,\
        type=int,\
        help="accetable deviance (in days) around sample date for USGSI, ITV, and ANA sites")
    parser.add_argument('--cloud_thr',\
        default=80,\
        type=int,\
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',\
        default=500,\
        type=int,\
        help="search radius used for reflectance data aggregation")
    parser.add_argument('--out_filetype',\
        default="csv",\
        type=str,\
        help="filetype for saved merged dataframe (csv or json)")
    parser.add_argument('--mask_method',\
        default="lulc",\
        type=str,\
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    args = parser.parse_args()

    ############### Setup ####################
    # arguments
    chip_size = args.buffer_distance
    day_tolerance = args.day_tolerance
    cloud_thr = args.cloud_thr
    out_filetype = args.out_filetype
    mask_method = args.mask_method

    # Set storage options for Azure blob storage
    with open("credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(' = ')
        os.environ[key] = value

    storage_options = {'account_name':os.environ['ACCOUNT_NAME'],
                    'account_key':os.environ['BLOB_KEY']}

    try:
        filepath = f"az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mask_method}_masking.csv"
        data = pd.read_csv(filepath, storage_options=storage_options)
    except:
        print(f"Error: no file at {filepath}")

    ## Add variables for stratifying data partition
    # SSC Quartile
    ssc = np.array(data["SSC (mg/L)"])
    ssc_quantiles = np.quantile(ssc, [0, 0.25, 0.5, 0.75])
    ssc_quantile_bin = np.digitize(ssc, ssc_quantiles)

    # year
    year = [dt.date.fromisoformat(i).year for i in list(data["Date-Time"])]

    # day of year
    julian = [dt.date.fromisoformat(i).timetuple().tm_yday for i in list(data["Date-Time"])]

    # add columns back to data
    data["julian"] = julian
    data["SSC Quantile"] = ssc_quantile_bin
    data["Year"] = year
    data["Season"] = np.digitize(np.array(data["julian"]), 366/2 * np.array([0, 1]))
    data["sine_julian"] = np.sin(2*np.pi*data["julian"]/365)
    data["is_brazil"] = 0
    data.loc[data["data_src"].isin(["itv", "ana"]), "is_brazil"] = 1

    ## Partition the data into train, test, validate
    # First split data into groups to ensure stratified
    grouped = data.groupby(by = ["SSC Quantile", "data_src"], group_keys=False)
    # now apply the train_test_validate_split function to each group
    partitioned = grouped.apply(lambda x: train_test_validate_split(x, [0.7, 0.15, 0.15]))

    out_filepath = f"az://modeling-data/partitioned_feature_data_buffer{chip_size}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{args.mask_method}_masking.{out_filetype}"

    if out_filetype == "csv":
        partitioned.to_csv(out_filepath, storage_options=storage_options)
    elif out_filetype == "json":
        partitioned.to_json(out_filepath, storage_options=storage_options)
    
    print(f"Done. Outputs written to {out_filepath}")
