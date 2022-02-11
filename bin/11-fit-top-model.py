import os, sys, pandas as pd, pickle, copy, argparse, json
sys.path.append("/content")
from src.utils import fit_mlp_full, MultipleRegression

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mse_to_minimize',
        default="mean_mse",
        choices=["mean_mse", "val_site_mse", "val_pooled_mse"],
        help="Which MSE to use, mean of sites, pooled, or the mean of the pooled and mean site MSE?"
    )
    args = parser.parse_args()

    results = pd.read_csv("output/mlp/grid_search_metadata_v3.csv")

    if not args.use_metadata_features:
        no_azimuth = ["mean_viewing_azimuth" not in results["features"][i] for i in range(0, (results.shape[0]))]
        results = results.loc[no_azimuth, :]

    results["mean_mse"] = (results["val_site_mse"] + results["val_pooled_mse"]) / 2

    # Eventually need to load this from blob storage, but pickle gives errors
    # Need to try switching to dill once we do the next grid search
    with open(results.iloc[results[args.mse_to_minimize].argmin(), :]["path"], "rb") as f:
        top_model = json.load(f)

    
    model_out = fit_mlp_full(
        features=top_model["features"],
        learning_rate=top_model["learning_rate"],
        batch_size=top_model["batch_size"],
        epochs=top_model["epochs"],
        storage_options=storage_options,
        activation_function=eval(f'nn.{top_model["activation"]}'),
        day_tolerance=top_model["day_tolerance"],
        cloud_thr=top_model["cloud_thr"],
        mask_method1="lulc",
        mask_method2="mndwi",
        min_water_pixels=20,
        layer_out_neurons=top_model["layer_out_neurons"],
        weight_decay=top_model["weight_decay"],
        verbose=True,
        model_out=f"mlp/top_model_metadata_{args.mse_to_minimize}"
    )