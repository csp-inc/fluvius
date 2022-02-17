### Environment setup
import sys, os
sys.path.append('/content')
from src.fluvius import WaterData
import fsspec
import pandas as pd
import argparse
import numpy as np
import torch, pickle, fsspec, torch.nn as nn
from src.utils import MultipleRegression
from sklearn.preprocessing import MinMaxScaler
import faulthandler
faulthandler.enable()

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
env_vars = open("/content/credentials","r").read().split('\n')

for var in env_vars[:-1]:
    key, value = var.split(' = ')
    os.environ[key] = value

storage_options= {'account_name':os.environ['ACCOUNT_NAME'],
                  'account_key':os.environ['BLOB_KEY']}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
        type=str,
        help="The path to the model state file (.pt file) without the subscript.")
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
        help="search radius used for reflectance data aggregation")
    parser.add_argument('--mask_method1',
        default="lulc",
        choices=["lulc", "scl"],
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',
        default="mndwi",
        choices=["ndvi", "mndwi", ""],
        type=str,
        help="Which additional index, if any, to use to update the mask, (\"ndvi\") or (\"mndwi\")")
    parser.add_argument('--n_folds',
        default=5,
        type=int,
        help="The number of folds to create for the training / validation set")
    parser.add_argument('--seed',
        default=123,
        type=int,
        help="The seed (an integer) used to initialize the pseudorandom number generator")
    args = parser.parse_args()
    day_tolerance = 0

    cloud_thr = args.cloud_thr
    buffer_distance = args.buffer_distance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    n_folds = args.n_folds
    seed = args.seed

    model_path = f"mlp/top_model_metadata_{args.mse_to_minimize}_{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}_v1"
    # Load in the top model metadata
    with open(f"{model_path}_metadata.pickle", "rb") as f:
        meta = pickle.load(f)

    model = MultipleRegression(len(meta["features"]), len(meta["layer_out_neurons"]), meta["layer_out_neurons"], activation_function=eval(f'nn.{meta["activation"]}'))

    with open(f"{model_path}.pt", "rb") as f:
        model.load_state_dict(torch.load(f))

    pred_fileprefix = f"az://prediction-data/{args.data_src}-data/feature_data_buffer{buffer_distance}m_daytol0_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking"

    pred_features = pd.read_csv(f"{pred_fileprefix}.csv", storage_options=storage_options)
    pred_features["is_brazil"] = 1
    
    data = pd.read_csv(meta["training_data"])
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