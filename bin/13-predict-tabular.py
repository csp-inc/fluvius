### Environment setup
import sys, os
sys.path.append('/content')
import json
import pandas as pd
import argparse
import numpy as np
import torch, fsspec, torch.nn as nn
from src.utils import MultipleRegression
from sklearn.preprocessing import MinMaxScaler
import faulthandler
faulthandler.enable()
from src.defaults import args_info

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
env_vars = open("/content/credentials","r").read().split('\n')

for var in env_vars[:-1]:
    key, value = var.split(' = ')
    os.environ[key] = value

storage_options= {'account_name':os.environ['ACCOUNT_NAME'],
                  'account_key':os.environ['BLOB_KEY']}

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src',
        type=args_info["data_src"]["type"],
        choices=args_info["data_src"]["choices"],
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

    fs = fsspec.filesystem("az", **storage_options)
    model_path = f"model-output/top_model_buffer{buffer_distance}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}"

    # Load in the top model metadata
    with fs.open(f"{model_path}_metadata.json", "r") as f:
        meta = json.load(f)

    model = MultipleRegression(len(meta["features"]), meta["layer_out_neurons"], activation_function=eval(f'nn.{meta["activation"]}'))

    with fs.open(f"{model_path}.pt", "rb") as f:
        model.load_state_dict(torch.load(f))

    pred_fileprefix = f"az://prediction-data/{args.data_src}-data/feature_data_buffer{buffer_distance}m_daytol0_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking"

    pred_features = pd.read_csv(f"{pred_fileprefix}.csv", storage_options=storage_options)
    pred_features["is_brazil"] = 1
    
    data = pd.read_csv(meta["training_data"], storage_options=storage_options)
    data["Log SSC (mg/L)"] = np.log(data["SSC (mg/L)"])

    data = data[data["partition"] != "testing"]

    response = "Log SSC (mg/L)"
    not_enough_water = data["n_water_pixels"] < meta["min_water_pixels"]
    data.drop(not_enough_water[not_enough_water].index, inplace=True)
    lnssc_0 = data["Log SSC (mg/L)"] == 0
    data.drop(lnssc_0[lnssc_0].index, inplace=True)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(data[meta["features"]])

    X_pred_scaled = np.array(scaler.transform(pred_features[meta["features"]]), dtype=float)
    scaled_pred_features = torch.tensor(X_pred_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(scaled_pred_features).squeeze().numpy().tolist()
    
    pred_features["Predicted Log SSC (mg/L)"] = y_pred

    pred_features.to_csv(f"az://predictions/{args.data_src}-predictions/feature_data_buffer{buffer_distance}m_daytol0_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv", storage_options=storage_options)