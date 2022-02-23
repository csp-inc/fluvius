import os, pandas as pd, argparse, pickle, fsspec, json
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
