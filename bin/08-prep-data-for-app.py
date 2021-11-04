from numpy import integer
import pandas as pd
import os
import fsspec
import datetime as dt, time
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
    parser.add_argument('--mask_method1',\
        default="lulc",\
        type=str,\
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',\
        default="",\
        type=str,\
        help="Which additional index to use to update the mask, (\"ndvi\") or (\"mndwi\")")
    args = parser.parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mm1 = args.mask_method1
    mm2 = args.mask_method2

    ### Prep JSON for app and cop chips to app storage container
    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}

    composites = [
        'rgb',
        'cir',
        'swir'
    ]

    fs = fsspec.filesystem("az", **storage_options)

    # Remove old app image chips if they exist
    for composite in composites:
        try:
            fs.rm(
                f"app/img/{composite}_and_{mm1}{mm2}_water", recursive=True
            )
        except FileNotFoundError:
            print(
                f"No directory exists at az://app/img/{composite}_" + \
                    f"and_{mm1}{mm2}_water, continuing..."
            )

        # Copy over new image chips -- need for-loop to avoid error
        paths_full = fs.ls(
            f"modeling-data/chips/qa/{composite}_{chip_size}m_" + \
                f"cloudthr{cloud_thr}_{mm1}{mm2}_masking/"
        )
        paths = [os.path.basename(x) for x in paths_full]

        for path in paths:
            fs.copy(
                f"modeling-data/chips/qa/{composite}_{chip_size}m_" + \
                    f"cloudthr{cloud_thr}_{mm1}{mm2}_masking/{path}",
                f"app/img/{composite}_and_{mm1}{mm2}_water/{path}"
            )

    # Read in metadata and prep JSON data for app
    site_metadata = pd.read_csv(
        "az://app/station_metadata.csv",
        storage_options=storage_options)
    all_data = pd.read_csv(
        f"az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer" +
        f"{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv",
        storage_options=storage_options
    )
    sites = all_data.site_no.unique()

    out_dicts = []
    url_base = "https://fluviusdata.blob.core.windows.net/app/img"
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
                "sample_timestamp": round(time.mktime(dt.date.fromisoformat(row["Date-Time_Remote"]).timetuple()))*1000,
                # "sample_julian": str(dt.date.fromisoformat(row["Date-Time"]).timetuple().tm_yday),
                # "acquisition_date": row["Date-Time_Remote"],
                # "acquisition_julian": str(dt.date.fromisoformat(row["Date-Time_Remote"]).timetuple().tm_yday),
                # "sentinel.2.l2a_R": str(round(row["sentinel-2-l2a_B04"])),
                # "sentinel.2.l2a_G": str(round(row["sentinel-2-l2a_B03"])),
                # "sentinel.2.l2a_B": str(round(row["sentinel-2-l2a_B02"])),
                # "sentinel.2.l2a_NIR": str(round(row["sentinel-2-l2a_B08"])),
                # "Chip.Cloud.Pct": str(round(row["Chip Cloud Pct"])),
                "rgb_water_chip_href":
                    f'{url_base}/rgb_and_{mm1}{mm2}_water/' + \
                        f'{os.path.basename(row["rgb_and_water_png_href"])}',
                "cir_water_chip_href":
                    f'{url_base}/cir_and_{mm1}{mm2}_water/' + \
                        f'{os.path.basename(row["cir_and_water_png_href"])}',
                "swir_water_chip_href":
                    f'{url_base}/swir_and_{mm1}{mm2}_water/' + \
                        f'{os.path.basename(row["swir_and_water_png_href"])}',
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
