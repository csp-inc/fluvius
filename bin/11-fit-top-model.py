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
    parser.add_argument('--cloud_thr',
        default=80,
        type=int,
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',
        default=500,
        type=int,
        help="search radius to use for reflectance data aggregation")
    parser.add_argument('--mask_method1',
        default="lulc",
        choices=["lulc", "scl"],
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',
        default="mndwi",
        choices=["ndvi", "mndwi", ""],
        type=str,
        help="Which additional index, if any, to use to update the mask, (\"ndvi\") or (\"mndwi\"), or \"\" to use no second mask")
    parser.add_argument('--n_folds',
        default=5,
        type=int,
        help="The number of folds to create for the training / validation set")
    parser.add_argument('--seed',
        default=123,
        type=int,
        help="The seed (an integer) used to initialize the pseudorandom number generator")
    args = parser.parse_args()
    cloud_thr = args.cloud_thr
    buffer_distance = args.buffer_distance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    n_folds = args.n_folds
    seed = args.seed

    results = pd.read_csv(f"az://model-output/mlp/grid_search_metadata_{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv", storage_options=storage_options)

    if not args.use_metadata_features:
        no_azimuth = ["mean_viewing_azimuth" not in results["features"][i] for i in range(0, (results.shape[0]))]
        results = results.loc[no_azimuth, :]

    results["mean_mse"] = (results["val_site_mse"] + results["val_pooled_mse"]) / 2

    # Eventually need to load this from blob storage, but pickle gives errors
    # Need to try switching to dill once we do the next grid search
    with open(results.iloc[results["mean_mse"].argmin(), :]["path"], "rb") as f:
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
        n_folds=n_folds,
        seed=seed,
        verbose=True
    )