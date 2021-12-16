if __name__ == "__main__":
    import os, sys, itertools
    sys.path.append("/content")
    from src.utils import fit_mlp_cv
    import multiprocessing as mp
    import pickle, hashlib, argparse, psutil
    import torch.nn as nn


    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers',
        default=psutil.cpu_count(logical = False),
        type=int,
        help="How many workers to use for fitting models in parallel (recommended not to go over number of physical cores"
    )
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
        help="Which additional index, if any, to use to update the mask, (\"ndvi\") or (\"mndwi\")")

    args = parser.parse_args()
    cloud_thr = args.cloud_thr
    buffer_distance = args.buffer_distance
    mm1 = args.mask_method1
    mm2 = args.mask_method2

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
            # Aerosol optical thickness
            "sentinel-2-l2a_AOT", 
            # RGB
            "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", "sentinel-2-l2a_B04",
            # Near infrared
            "sentinel-2-l2a_B08",
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
            "sentinel-2-l2a_B11", "sentinel-2-l2a_B12"
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
            "is_brazil",
            # Scene metadata
            "mean_viewing_azimuth", "mean_viewing_zenith",
            "mean_solar_azimuth", "mean_solar_zenith"
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
            "is_brazil",
            # Scene metadata
            "mean_viewing_azimuth", "mean_viewing_zenith",
            "mean_solar_azimuth", "mean_solar_zenith"
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
            "is_brazil",
            # Short-wave infrared
            "sentinel-2-l2a_B11", "sentinel-2-l2a_B12"
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
            # Scene metadata
            "mean_viewing_azimuth", "mean_viewing_zenith",
            "mean_solar_azimuth", "mean_solar_zenith"
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
            "is_brazil",
            # Short-wave infrared
            "sentinel-2-l2a_B11", "sentinel-2-l2a_B12",
            # Scene metadata
            "mean_viewing_azimuth", "mean_viewing_zenith",
            "mean_solar_azimuth", "mean_solar_zenith"
        ]
    ]

    epochs = [500, 1000, 1500]
    batch_size = [16, 32, 48, 64]
    learning_rate = [0.01, 0.005, 0.001]
    learn_sched_gamma = [0.5, 0.2]
    learn_sched_step = [200]


    layer_out_neurons = [
        [ 6, 12, 6],
        [ 6, 24, 6],
        [12, 16, 8],
        [12, 24, 8],
        [ 4,  8, 4],
        [12,  6, 3],
        [24, 12, 6]  
    ]

    activation = [nn.ReLU(), nn.SELU(), nn.PReLU(init=0.05)]

    permutations = list(
        itertools.product(
            features,
            learning_rate,
            batch_size,
            epochs,
            layer_out_neurons,
            learn_sched_gamma,
            learn_sched_step,
            activation
        )
    )
    print(len(permutations))
    if not os.path.exists(f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_tmp_5fold"):
        os.makedirs(f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_tmp_5fold")
    
    
    def fit_model(args):
        args_hash = hashlib.sha224("_".join([str(x) for x in args]).encode("utf-8")).hexdigest()[0:20]
        fn = f"output/mlp/{buffer_distance}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking_tmp_5fold/{args_hash}.pickle"

        if not os.path.exists(fn):
            model_out = fit_mlp_cv(
                features=args[0],
                learning_rate=args[1],
                batch_size=args[2],
                epochs=args[3],
                storage_options=storage_options,
                activation_function=args[7],
                day_tolerance=8,
                cloud_thr=cloud_thr,
                mask_method1=mm1,
                mask_method2="mndwi",
                min_water_pixels=20,
                layer_out_neurons=args[4],
                learn_sched_step_size=args[6],
                learn_sched_gamma=args[5],
                verbose=False
            )
            
            with open(fn, 'wb') as f:
                pickle.dump(model_out, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Model output already exists. Skipping...")

    print(f"Beginning model fits with {args.n_workers} workers in parallel...")        
  
    my_pool = mp.Pool(processes=args.n_workers)
    my_pool.map(fit_model, permutations, chunksize=4)
