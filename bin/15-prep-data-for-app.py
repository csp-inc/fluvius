from numpy import integer
import pandas as pd, numpy as np
import os
import fsspec
import datetime as dt, time
import json
import argparse
from src.defaults import args_info

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day-tolerance',
        default=8,
        type=int,
        help="accetable deviance (in days) around sample date for USGSI, ITV, and ANA sites")
    parser.add_argument('--cloud-thr',
        default=80,
        type=int,
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer-distance',
        default=500,
        type=int,
        help="search radius used for reflectance data aggregation")
    parser.add_argument('--mask-method1',
        default="lulc",
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask-method2',
        default="mndwi",
        type=str,
        help="Which additional index to use to update the mask, (\"ndvi\") or (\"mndwi\")")
    parser.add_argument('--n-folds',
        default=5,
        type=int,
        help="The number of folds to create for the training / validation set")
    parser.add_argument('--seed',
        default=123,
        type=int,
        help="The seed (an integer) used to initialize the pseudorandom number generator")
    return parser

if __name__ == "__main__":
    
    args = return_parser().parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    n_folds = args.n_folds
    seed = args.seed

    ### Prep JSON for app and cop chips to app storage container
    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}


    fs = fsspec.filesystem("az", **storage_options)

    # Read in metadata and prep JSON data for app
    site_metadata = pd.read_csv(
        "az://app/station_metadata.csv",
        storage_options=storage_options)
    all_data = pd.read_csv(
        f"az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer" +
        f"{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv",
        storage_options=storage_options
    )

    all_data = all_data.loc[all_data.region.isin(["itv", "ana"]), ]
    itv_predictions = pd.read_csv(f"az://predictions/itv-predictions/prediction_data_buffer{chip_size}m_daytol0_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv", storage_options=storage_options)
    ana_predictions = pd.read_csv(f"az://predictions/ana-predictions/prediction_data_buffer{chip_size}m_daytol0_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv", storage_options=storage_options)
    prediction_data = pd.concat([itv_predictions, ana_predictions])
    pred_sites = [x.split("_")[0] for x in prediction_data["sample_id"]]
    prediction_data["site_no"] = pred_sites
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
                "SSC.mg.L": str(round(row["SSC (mg/L)"], 2)),
                #"Q.m3.s": str(row["Q (m3/s)"]),
                "sample_date": row["Date-Time"],
                "timestamp": round(time.mktime(dt.date.fromisoformat(row["Date-Time_Remote"]).timetuple()))*1000,
                # "sample_julian": str(dt.date.fromisoformat(row["Date-Time"]).timetuple().tm_yday),
                # "acquisition_date": row["Date-Time_Remote"],
                # "acquisition_julian": str(dt.date.fromisoformat(row["Date-Time_Remote"]).timetuple().tm_yday),
                # "sentinel.2.l2a_R": str(round(row["sentinel-2-l2a_B04"])),
                # "sentinel.2.l2a_G": str(round(row["sentinel-2-l2a_B03"])),
                # "sentinel.2.l2a_B": str(round(row["sentinel-2-l2a_B02"])),
                # "sentinel.2.l2a_NIR": str(round(row["sentinel-2-l2a_B08"])),
                # "Chip.Cloud.Pct": str(round(row["Chip Cloud Pct"]))
                #"raw_img_chip": row["raw_img_chip_href"]
            })
        
        site_pred_df = prediction_data.loc[prediction_data["site_no"] == site, ].reset_index()
        predictions = []
        for i, row in site_pred_df.iterrows():
            predictions.append({
                "prediction_id": row["sample_id"],
                "SSC.mg.L": str(round(np.exp(row["Predicted Log SSC (mg/L)"]), 2)),
                "prediction_date": row["Date-Time"],
                "timestamp": round(time.mktime(dt.date.fromisoformat(row["Date-Time"]).timetuple()))*1000,
                "pred_chip": row["app_chip_href"]
            })

        out_dicts.append({
            "region": region,
            "site_no": site,
            "site_name": [x["site_name"] for i,x in site_metadata.iterrows() if x["site_no"].zfill(8) == site][0],
            "Longitude": lon,
            "Latitude": lat,
            "sample_data": samples,
            "predictions": predictions
        })

    # Push to final JSON data that app reads in
    with fs.open('app/all_data_v4.json', 'w') as fn:
        json.dump(out_dicts, fn)
