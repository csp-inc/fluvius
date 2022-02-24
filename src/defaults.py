# import psutil

args_info = {
    "get_instantaneous": {
        "action": "store_true", # ensures the default value is False
        "help": "Get instantaneous flow data or modeled continuous data"
    },
    "write_to_csv": {
        "action": "store_true", # ensures the default value is False
        "help": "Write out CSVs to ./data"
    },
    "index_start": {
        "default": 0,
        "type": int,
        "help": "Array indexing (0-based or 1-based)"
    },
    "data_src": {
        "type": str,
        "choices": ["itv", "ana", "usgs", "usgsi"],
        "help": "Data source code"
    },
    "day_tolerance": {
        "default": 8,
        "type": int,
        "help": "Days of search around sample date"
    },
    "cloud_thr": {
        "default": 80,
        "type": int,
        "help": "Percent of cloud cover acceptable"
    },
    "buffer_distance": {
        "default": 500,
        "type": int,
        "help": "Search radius to use for reflectance data aggregation"
    },
    "write_chips": {
        "action": "store_true", # ensures the default value is False
        "help": "Write chips to blob storage?"
    },
    "mask_method1": {
        "default": "lulc",
        "type": str,
        "choices": ["lulc", "scl"],
        "help": "Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")"
    },
    "mask_method2": {
        "default": "mndwi",
        "type": str,
        "choices": ["mndwi", "ndvi", ""],
        "help": "Which additional normalized index to use, if any, to update the mask"
    },
    "out_filetype": {
        "default": "csv",
        "type": str,
        "choices": ["csv", "json"],
        "help": "Filetype for saved merged dataframe (csv or json)"
    },
    "n_folds": {
        "default": 5,
        "type": int,
        "help": "The number of folds to create for the training / validation set"
    },
    "seed": {
        "default": 123,
        "type": int,
        "help": "The seed (an integer) used to initialize the pseudorandom number generator"
    },
    "write_qa_chips": {
        "action": "store_false", # ensures the default value is True
        "help": "Should QA chips be written to blob storage?"
    },
    "rgb_min": {
        "default": 100,
        "type": int,
        "help": "Minimum reflectance value (corresponding to zero saturation of R, G, and B)"
    },
    "rgb_max": {
        "default": 4000,
        "type": int,
        "help": "Maximum reflectance value (corresponding to full saturation of R, G, and B)"
    },
    "gamma": {
        "default": 0.7,
        "type": float,
        "help": "Gamma correction to use when generating the image chip"
    },
    "local_outpath": {
        "default": None,
        "type": str,
        "help": "If desired, the local directory to which QA chips will be saved. If \"None\", the default, chips are not written locally"
    },
    "composite": {
        "default": "rgb",
        "type": str,
        "choices": ["rgb", "cir", "swir"],
        "help": "Which color composite to download. \"rgb\", color infrared (\"cir\"), or short-wave infrared (\"swir\")"
    },
    "qa_chip_list_name": {
        "default": "good_chips.csv",
        "type": str,
        "help": "The file name for the list of good chips"
        # The Azure filename for the list of good chips. " +
        #      "Assumed to be in the \"modeling-data/chips/post-qa/{chip_size}" +
        #      "m_cloudthr{cloud_thr}_{mm1}{mm2}_masking` directory\". " +
        #      "This will be used to determine which samples in the feature " +
        #      "dataframe to keep.
    },
    "n_workers": {
        "default": 2, #psutil.cpu_count(logical = False),
        "type": int,
        "help": "How many workers to use for fitting models in parallel (recommended not to go over number of physical cores)"
    },
    "model_path": {
        "type": str,
        "help": "The path to the model state file (.pt file) without the subscript"
    },
    "start_date": {
        "default": "2015-01-01",
        "type": str,
        "help": "The earliest date for which to generate prediction inputs"
    },
    "end_date": {
        "default": "2021-12-31",
        "type": str,
        "help": "The latest date for which to generate prediction inputs"
    },
    "mse_to_minimize": {
        "default": "mean_mse",
        "type": str,
        "choices": ["mean_mse", "val_site_mse", "val_pooled_mse"],
        "help": "Which MSE to use, mean of sites, pooled, or the mean of the pooled and mean site MSE?"
    }
}