### Environment setup
import os
import pandas as pd
import argparse

COLUMNS = ["data_src", "sample_id", "Longitude", "Latitude", "Date-Time",
           "Date", "Date-Time_Remote", "SSC (mg/L)", "Q (m3/s)",
           "InSitu_Satellite_Diff", "Chip Cloud Pct", "sentinel-2-l2a_AOT",
           "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", "sentinel-2-l2a_B04",
           "sentinel-2-l2a_B08", "sentinel-2-l2a_WVP", "sentinel-2-l2a_B05",
           "sentinel-2-l2a_B06", "sentinel-2-l2a_B07", "sentinel-2-l2a_B8A",
           "sentinel-2-l2a_B11", "sentinel-2-l2a_B12", "n_water_pixels",
           "mean_viewing_azimuth", "mean_viewing_zenith", "mean_solar_azimuth",
           "mean_solar_zenith", "sensing_time"]

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
    parser.add_argument('--mask_method1',\
        default="lulc",\
        type=str,\
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',\
        default="",\
        type=str,\
        help="Which additional index to use to update the mask, (\"ndvi\") or (\"mndwi\")")
    args = parser.parse_args()
    ############### Setup ####################
    # arguments
    buffer_distance = args.buffer_distance
    day_tolerance = args.day_tolerance
    cloud_thr = args.cloud_thr
    out_filetype = args.out_filetype
    mm1 = args.mask_method1
    mm2 = args.mask_method2

    #storage options
    env_vars = open("/content/credentials","r").read().split('\n')

    for var in env_vars[:-1]:
        key, value = var.split(' = ')
        os.environ[key] = value

    storage_options={'account_name':os.environ['ACCOUNT_NAME'],\
                    'account_key':os.environ['BLOB_KEY']}

    ############### Read in and merge data ####################
    try:
        itv_path = f"az://modeling-data/itv-data/feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv"
        itv_data = pd.read_csv(itv_path, storage_options=storage_options)\
            .rename(columns={"SSC (mg/l)": "SSC (mg/L)",
                             "Q (m³/s)": "Q (m3/s)"})\
            .assign(data_src="itv")[COLUMNS].dropna()
    except:
        print(f"Error: Bad filepath: {itv_path}")
        print(f"No ITV dataset with buffer {args.buffer_distance}m, day tolerance {args.day_tolerance}, masking method {args.mask_method}, and cloud threshold {args.cloud_thr}")
        quit()
    try:
        ana_path = f"az://modeling-data/ana-data/feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{args.mask_method}_masking.csv"
        ana_data = pd.read_csv(ana_path, storage_options=storage_options)\
            .rename(columns={
                "Suspended Sediment Concentration (mg/L)": "SSC (mg/L)",
                "Discharge": "Q (m3/s)"})\
            .assign(data_src="ana")[COLUMNS].dropna()
    except:
        print(f"Error: Bad filepath: {ana_path}")
        print(f"No ANA dataset with buffer {args.buffer_distance}m, day tolerance {args.day_tolerance}, masking method {args.mask_method}, and cloud threshold {args.cloud_thr}")
        quit()
    try:
        usgsi_path = f"az://modeling-data/usgsi-data/feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{args.mask_method}_masking.csv"
        usgsi_data = pd.read_csv(usgsi_path, storage_options=storage_options)\
            .assign(cms=lambda x: x["Instantaneous computed discharge (cfs)_x"] / 35.314666721488588)\
            .rename(
                columns={
                    "Instantaneous suspended sediment (mg/L)": "SSC (mg/L)",
                    "cms": "Q (m3/s)"
                }
            )\
            .assign(data_src="usgsi")[COLUMNS].dropna()
    except:
        print(f"Error: Bad filepath: {usgsi_path}")
        print(f"No USGSI dataset with buffer {args.buffer_distance}m, day tolerance {args.day_tolerance}, masking method {args.mask_method}, and cloud threshold {args.cloud_thr}")
        quit()
    try:
        usgs_path = f"az://modeling-data/usgs-data/feature_data_buffer{buffer_distance}m_daytol0_cloudthr{cloud_thr}percent_{args.mask_method}_masking.csv"
        usgs_data = pd.read_csv(usgs_path, storage_options=storage_options)\
            .assign(cms=lambda x: x["Instantaneous computed discharge (cfs)"] / 35.314666721488588)\
            .rename(
                columns={
                    "Computed instantaneous suspended sediment (mg/L)": "SSC (mg/L)",
                    "cms": "Q (m3/s)"
                }
            ).assign(data_src="usgs")[COLUMNS].dropna()
    except:
        print(f"Error: Bad filepath: {usgs_path}")
        print(f"No USGS dataset with buffer {args.buffer_distance}m, masking method {args.mask_method}, and cloud threshold {args.cloud_thr}")
        quit()

    merged_df = pd.concat([itv_data, ana_data, usgsi_data, usgs_data])

    # Drop "problem" observations
    bad_ssc = (merged_df["SSC (mg/L)"] == "--") |\
              (merged_df["SSC (mg/L)"] == 0) |\
              (merged_df["SSC (mg/L)"] == "0.0")

    merged_df.drop(bad_ssc[bad_ssc].index, inplace=True)

    # NOTE this is very hacky right now -- should revise/improve
    bad_rgb = ((merged_df["sentinel-2-l2a_B02"] == 0) &\
              (merged_df["sentinel-2-l2a_B03"] == 0) &\
              (merged_df["sentinel-2-l2a_B04"] == 0)) |\
                  ((merged_df["sentinel-2-l2a_B02"] > 15000) &\
                   (merged_df["sentinel-2-l2a_B03"] > 15000) &\
                   (merged_df["sentinel-2-l2a_B04"] > 15000))

    merged_df.drop(bad_rgb[bad_rgb].index, inplace=True)
    # write output
    out_filepath = f"az://modeling-data/merged_feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{args.mask_method}_masking.{out_filetype}"
    if out_filetype == "csv":
        merged_df.to_csv(out_filepath, storage_options=storage_options)
    elif out_filetype == "json":
        merged_df.to_json(out_filepath, storage_options=storage_options)

    print(f"Done. Outputs written to {out_filepath}")
