import os, sys, pandas as pd, pickle, copy, argparse, json
sys.path.append("/content")
from src.utils import fit_mlp_full
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
    cloud_thr = args.cloud_thr # TODO: talk to Vincent, I'm not sure these make sense if they are 'inherited' from the top model
    buffer_distance = args.buffer_distance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    n_folds = args.n_folds
    seed = args.seed

    results = pd.read_csv(f"az://model-output/mlp/grid_search_metadata_{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv", storage_options=storage_options)

    # Compute the overall mean MSE as the average of the site MSE (mean of the site mean MSEs) and the pooled MSE.
    results["mean_mse"] = (results["val_site_mse"] + results["val_pooled_mse"]) / 2

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
        mask_method1=mm1,
        mask_method2=mm2,
        min_water_pixels=20,
        layer_out_neurons=top_model["layer_out_neurons"],
        weight_decay=top_model["weight_decay"],
        n_folds=n_folds,
        seed=seed,
        verbose=True
    )