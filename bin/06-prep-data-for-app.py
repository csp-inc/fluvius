import pandas as pd
import os
import fsspec
import datetime as dt
import json
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

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mask_method = args.mask_method

    ### Prep JSON for app and cop chips to app storage container
    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}

    fs = fsspec.filesystem("az", **storage_options)

    # Remove old app image chips if they exist
    try:
        fs.rm(f"app/img/rgb_and_{mask_method}_water", recursive=True)
    except FileNotFoundError:
        print(f"No directory exists at az://app/img/rgb_and{mask_method}_water, continuing...")

    # Copy over new image chips -- need for-loop to avoid error 
    for path in [os.path.basename(x) for x in fs.ls(f"modeling-data/chips/qa/rgb_{chip_size}m_cloudthr{cloud_thr}_{mask_method}_masking/")]:
        fs.copy(
            f"modeling-data/chips/qa/rgb_{chip_size}m_cloudthr{cloud_thr}_{mask_method}_masking/{path}",
            f"app/img/rgb_and_{mask_method}_water/{path}"
        )

    # Read in metadata and prep JSON data for app
    site_metadata = pd.read_csv("az://app/station_metadata.csv", storage_options=storage_options)
    all_data = pd.read_csv(f"az://modeling-data/fluvius_data_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mask_method}_masking.csv", storage_options=storage_options)
    sites = all_data.site_no.unique()
    
    out_dicts = []
    for site in sites:
        site_df = all_data[all_data["site_no"] == site].reset_index()
        region = site_df.region.iloc[0]
        lat = site_df.Latitude.iloc[0]
        lon = site_df.Longitude.iloc[0]

        samples = []
        for i, row in site_df.iterrows():
            samples.append({
                "sample_id": row["sample_id"],
                "SSC.mg.L": str(row["SSC (mg/L)"]),
                "Q.m3.s": str(row["Q (m3/s)"]),
                "sample_date": row["Date-Time"],
                # "sample_julian": str(dt.date.fromisoformat(row["Date-Time"]).timetuple().tm_yday),
                # "acquisition_date": row["Date-Time_Remote"],
                # "acquisition_julian": str(dt.date.fromisoformat(row["Date-Time_Remote"]).timetuple().tm_yday),
                # "sentinel.2.l2a_R": str(round(row["sentinel-2-l2a_B04"])),
                # "sentinel.2.l2a_G": str(round(row["sentinel-2-l2a_B03"])),
                # "sentinel.2.l2a_B": str(round(row["sentinel-2-l2a_B02"])),
                # "sentinel.2.l2a_NIR": str(round(row["sentinel-2-l2a_B08"])),
                # "Chip.Cloud.Pct": str(round(row["Chip Cloud Pct"])),
                "rgb_water_chip_href": f'https://fluviusdata.blob.core.windows.net/app/img/rgb_and_{mask_method}_water/{os.path.basename(row["rgb_and_water_png_href"])}'
                #"raw_img_chip": row["raw_img_chip_href"]
            })
        
        out_dicts.append({
            "region": row["region"],
            "site_no": site,
            "site_name": [x["site_name"] for i,x in site_metadata.iterrows() if x["site_no"].zfill(8) == site][0],
            "Longitude": lon,
            "Latitude": lat,
            "sample_data": samples
        })

    # Push to final JSON data that app reads in
    with fs.open('app/all_data_v2.json', 'w') as fn:
        json.dump(out_dicts, fn)