import os, sys
sys.path.append("/content")
import json, argparse
from src.utils import MultipleRegression, predict_chip, RGB_MIN, RGB_MAX, GAMMA
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn
import pandas as pd, numpy as np, rasterio as rio
from PIL import Image
import fsspec
from src.defaults import args_info

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src',
        type=args_info["data_src"]["type"],
        choices=args_info["data_src"]["choices"], # NOTE: just ["itv", "ana"]?
        help=args_info["data_src"]["help"])
    parser.add_argument('--cloud-thr',
        default=args_info["cloud_thr"]["default"],
        type=args_info["cloud_thr"]["type"],
        help=args_info["cloud_thr"]["help"])
    parser.add_argument('--buffer-distance',
        default=args_info["buffer_distance"]["default"],
        type=args_info["buffer_distance"]["type"],
        help=args_info["buffer_distance"]["help"])
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
    parser.add_argument('--n-folds',
        default=args_info["n_folds"]["default"],
        type=args_info["n_folds"]["type"],
        help=args_info["n_folds"]["help"])
    parser.add_argument('--seed',
        default=args_info["seed"]["default"],
        type=args_info["seed"]["type"],
        help=args_info["seed"]["help"])
    return parser

if __name__ == "__main__":
    
    args = return_parser().parse_args()
    day_tolerance = 0
    cloud_thr = args.cloud_thr
    buffer_distance = args.buffer_distance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    n_folds = args.n_folds
    seed = args.seed

    mm1 = args.mask_method1
    mm2 = args.mask_method2

    # Create filesystem
    fs = fsspec.filesystem("az", **storage_options)

    model_path =  f"model-output/top_model_buffer{buffer_distance}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}"

    # Load in the top model metadata
    with fs.open(f"{model_path}_metadata.json", "rb") as f:
        meta = json.load(f)

    model = MultipleRegression(len(meta["features"]), len(meta["layer_out_neurons"]), meta["layer_out_neurons"], activation_function=eval(f'nn.{meta["activation"]}'))
    with fs.open(f"{model_path}.pt", "rb") as f:
        model.load_state_dict(torch.load(f))
    
    # Wrangle the training data to recreate the scaler and get info on features
    fp = meta["training_data"]
    data = pd.read_csv(fp, storage_options=storage_options)

    features = meta["features"]

    not_enough_water = data["n_water_pixels"] <= meta["min_water_pixels"]
    data.drop(not_enough_water[not_enough_water].index, inplace=True)
    data["Log SSC (mg/L)"] = np.log(data["SSC (mg/L)"])
    data["is_brazil"] = [float(x) for x in data["is_brazil"]]
    lnssc_0 = data["Log SSC (mg/L)"] == 0
    data.drop(lnssc_0[lnssc_0].index, inplace=True)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(data[meta["features"]])

    sentinel_features = [x for x in features if "sentinel" in x]
    non_sentinel_features = [x for x in features if "sentinel" not in x]

    # Load feature data for making extrapolations, add hrefs as columns
    fp_pred = f"az://predictions/{args.data_src}-predictions/feature_data_buffer{buffer_distance}m_daytol0_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv"

    pred_df = pd.read_csv(fp_pred, storage_options=storage_options).dropna()
    if args.data_src in ["itv", "ana"]:
        pred_df["is_brazil"] = 1
    else:
        pred_df["is_brazil"] = 0

    # Keep only prediction site chips that have 20 or more water pixels
    # Only keep prediction site chips that have < 50% clouds unless they have a corresponding observation in the training data
    pred_df["has_obs"] = [pred_df["Date-Time"].iloc[i] in list(data.loc[data["site_no"] == pred_df["sample_id"].iloc[i][0:8], ]["Date-Time_Remote"]) for i in range(0, len(pred_df))]
    pred_df = pred_df.loc[(pred_df["n_water_pixels"] >= meta["min_water_pixels"]) & ((pred_df["has_obs"] == True) | (pred_df["Chip Cloud Pct"] <= 50)), ]

    # Generate hrefs for the image chips for prediction sites
    raw_img_hrefs = []
    raw_water_hrefs = []

    href_base = f"https://fluviusdata.blob.core.windows.net/prediction-data/chips/{args.buffer_distance}m_cloudthr{args.cloud_thr}_{mm1}{mm2}_masking/{args.data_src}/"
    for _, row in pred_df.iterrows():
        fn_root = f"{row['sample_id']}_{row['Date-Time']}"
        raw_img_hrefs.append(f"{href_base}{fn_root}.tif")
        raw_water_hrefs.append(f"{href_base}{fn_root}_water.tif")
        
    pred_df["raw_img_chip_href"] = raw_img_hrefs
    pred_df["water_chip_href"] = raw_water_hrefs

    # Using the new DataFrame, create chips of pixel-wise SSC
    app_chip_hrefs = []
    app_chip_fn_base = f"app/img/prediction_chips_{args.buffer_distance}m_cloudthr{args.cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}/"
    for _, row in pred_df.reset_index().iterrows():
        pred_chip = predict_chip(features, sentinel_features, non_sentinel_features, row, model, scaler)

        ## Create merged image for app display
        with rio.Env(
            AZURE_STORAGE_ACCOUNT=os.environ["ACCOUNT_NAME"],
            AZURE_STORAGE_ACCESS_KEY=os.environ["BLOB_KEY"]
        ):
            with rio.open(f"az://{row['raw_img_chip_href'][42:]}") as chip:
                rgb_raw = chip.read((4, 3, 2))

        rgb = np.moveaxis(
            np.interp(
                np.clip(
                    rgb_raw,
                    RGB_MIN,
                    RGB_MAX
                ), 
                (RGB_MIN, RGB_MAX),
                (0, 1)
            ) ** GAMMA * 255,
            0,
            2
        ).astype(np.uint8)
        
        app_chip = np.concatenate([rgb, np.full((rgb.shape[0], 5, rgb.shape[2]), 0).astype(np.uint8), pred_chip], axis = (1))
        app_fn = f"{app_chip_fn_base}{args.data_src}_{row['sample_id']}_{row['Date-Time']}_chip.png"
        with fs.open(app_fn, "wb") as fn:
            Image.fromarray(app_chip).save(fn, "PNG")

        app_chip_hrefs.append(f"https://fluviusdata.blob.core.windows.net/{app_fn}")

    pred_df["app_chip_href"] = app_chip_hrefs

    # save final prediction data with hrefs to Azure
    cols = ['sample_id', 'Longitude', 'Latitude', 'Date-Time', 'Date-Time_Remote', 
       'Tile Cloud Cover', 'InSitu_Satellite_Diff',
       'Chip Cloud Pct', 'sentinel-2-l2a_AOT', 'sentinel-2-l2a_B02',
       'sentinel-2-l2a_B03', 'sentinel-2-l2a_B04', 'sentinel-2-l2a_B08',
       'sentinel-2-l2a_WVP', 'sentinel-2-l2a_B05', 'sentinel-2-l2a_B06',
       'sentinel-2-l2a_B07', 'sentinel-2-l2a_B8A', 'sentinel-2-l2a_B11',
       'sentinel-2-l2a_B12', 'n_water_pixels',
       'sensing_time', 'is_brazil', 'Predicted Log SSC (mg/L)', 'has_obs',
       'raw_img_chip_href', 'water_chip_href', 'app_chip_href']

    pred_df.reset_index()[cols].to_csv(f"az://predictions/{args.data_src}-predictions/prediction_data_buffer{args.buffer_distance}m_daytol0_cloudthr{args.cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv", storage_options=storage_options)
