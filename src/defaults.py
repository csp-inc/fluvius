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
    }
}