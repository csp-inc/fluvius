import os, sys, itertools
sys.path.append("/content")
from src.utils import fit_mlp_cv
import multiprocessing as mp
import pickle, hashlib, argparse, psutil
import torch.nn as nn
import json
import numpy as np
from src.defaults import args_info
    
def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers',
        default=args_info["n_workers"]["default"],
        type=args_info["n_workers"]["type"],
        help=args_info["n_workers"]["help"])
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

    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}
    
    #### Set possible values for each argument
    buffer_distance = 500
    day_tolerance = 8
    cloud_thr = 80
    min_water_pixels = 20

    features = [
        [
            # RGB
            "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", "sentinel-2-l2a_B04",
            # Near infrared
            "sentinel-2-l2a_B08",
            # Red edge bands
            "sentinel-2-l2a_B07", "sentinel-2-l2a_B8A",
            "sentinel-2-l2a_B05", "sentinel-2-l2a_B06",
            # Site/time variables
            "is_brazil"
        ],
        [
            # Aerosol optical thickness
            "sentinel-2-l2a_AOT", 
            # RGB
            "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", "sentinel-2-l2a_B04",
            # Near infrared
            "sentinel-2-l2a_B08",
            # Red edge bands
            "sentinel-2-l2a_B07", "sentinel-2-l2a_B8A",
            "sentinel-2-l2a_B05", "sentinel-2-l2a_B06",
            # Site/time variables
            "is_brazil"
        ],
        [
            # RGB
            "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", "sentinel-2-l2a_B04",
            # Near infrared
            "sentinel-2-l2a_B08",
            # Red edge bands
            "sentinel-2-l2a_B07", "sentinel-2-l2a_B8A",
            "sentinel-2-l2a_B05", "sentinel-2-l2a_B06",
            # Short-wave infrared
            "sentinel-2-l2a_B11", "sentinel-2-l2a_B12",
            # Site/time variables
            "is_brazil"
        ],
        [
            # Aerosol optical thickness
            "sentinel-2-l2a_AOT", 
            # RGB
            "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", "sentinel-2-l2a_B04",
            # Near infrared
            "sentinel-2-l2a_B08",
            # Red edge bands
            "sentinel-2-l2a_B07", "sentinel-2-l2a_B8A",
            "sentinel-2-l2a_B05", "sentinel-2-l2a_B06",
            # Short-wave infrared
            "sentinel-2-l2a_B11", "sentinel-2-l2a_B12",
            # Site/time variables
            "is_brazil"
        ]
    ]

    epochs = [15000]
    batch_size = [16, 32]
    learning_rate = [0.0001, 0.0005]

    layer_out_neurons = [
        [5, 3, 2],
        [4, 4, 2],
        [4, 2, 2],
        [6, 5],
        [6, 3]
    ]

    activation = [nn.PReLU(num_parameters=1), nn.SELU()]
    weight_decay = [0, 1e-2]
    permutations = list(
        itertools.product(
            features,
            learning_rate,
            batch_size,
            epochs,
            layer_out_neurons,
            weight_decay,
            activation,
        )
    )
    print(f"Fitting {len(permutations)} models...")
    if not os.path.exists(f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}"):
        os.makedirs(f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}")
    
    
    def fit_model(args):
        args_hash = hashlib.sha224("_".join([str(x) for x in args]).encode("utf-8")).hexdigest()[0:20]
        fn = f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}/{args_hash}.json"

        if not os.path.exists(fn):
            model_out = fit_mlp_cv(
                features=args[0],
                learning_rate=args[1],
                batch_size=args[2],
                epochs=args[3],
                storage_options=storage_options,
                activation_function=args[6],
                day_tolerance=8,
                cloud_thr=cloud_thr,
                mask_method1=mm1,
                mask_method2="mndwi",
                min_water_pixels=20,
                layer_out_neurons=args[4],
                weight_decay=args[5],
                verbose=False
            )
            
            with open(fn, 'w') as f:
                json.dump(model_out, f)
        else:
            print("Model output already exists. Skipping...")

    n_workers = args.n_workers
    if np.isnan(n_workers):
        n_workers = psutil.cpu_count(logical = False)

    print(f"Beginning model fits with {n_workers} workers in parallel...")        

    my_pool = mp.Pool(processes=n_workers)
    my_pool.map(fit_model, permutations, chunksize=4)
