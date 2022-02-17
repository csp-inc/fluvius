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
    # parser.add_argument('--path_to_models',
    #     default="output/mlp/500m_cloudthr80_lulcmndwi_masking_{n_folds}_folds_seed{seed}_v1",
    #     type=str,
    #     help="Path to the model outputs")
    parser.add_argument('--results_output',
        default="mlp/grid_search_metadata_v1.csv",
        type=str,
        help="Where shoul results be saved? Path should be relative to the output folder locally and the model-output container on blob storage.")
    parser.add_argument('--n_folds',
        default=5,
        type=int,
        help="The number of folds to create for the training / validation set")
    parser.add_argument('--seed',
        default=123,
        type=int,
        help="The seed (an integer) used to initialize the pseudorandom number generator")
    args = parser.parse_args()

    directory = f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}_folds_seed{seed}_v1" # was: args.path_to_models
    
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
    metadata.to_csv(f"az://model-output/{args.results_output}", index=False, storage_options=storage_options)
