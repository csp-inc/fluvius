import os, pandas as pd, argparse, pickle, fsspec, json

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

if __name__ == "__main__":
    ############### Parse commnd line args ###################
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

    directory = f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}_v1" # was: args.path_to_models
    
    paths = [f"{directory}/{x}" for x in os.listdir(directory)]

    metadata = pd.DataFrame(
        columns=[
            "id", "path", "features", "layers", "activation_layer", "epochs",
            "batch_size", "weight_decay", "val_site_mse", "val_pooled_mse"
        ]
    )
    for path in paths:
        with open(path, "r") as f:
            model_dict = json.load(f)

        row_dict = {
            "id": os.path.basename(path)[:-7],
            "path": path,
            "features": str(model_dict["features"]),
            "layers": model_dict["layer_out_neurons"],
            "activation_layer": model_dict["activation"],
            "epochs": model_dict["epochs"],
            "learning_rate": model_dict["learning_rate"],
            "batch_size": model_dict["batch_size"],
            "weight_decay": model_dict["weight_decay"],
            "val_site_mse": model_dict["val_site_mse"],
            "val_pooled_mse": model_dict["val_pooled_mse"]
        }

        metadata = metadata.append(row_dict, ignore_index=True)
    
    metadata.to_csv(f"output/{args.results_output}", index=False)
    metadata.to_csv(f"az://model-output/mlp/grid_search_metadata_{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}_v1.csv", index=False, storage_options=storage_options)
