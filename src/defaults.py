import numpy as np

args_info = {
    "get_instantaneous": {
        "action": "store_true", # ensures the default value is False
        "help": "Get instantaneous flow data or modeled continuous data."
    },
    "write_to_csv": {
        "action": "store_true", # ensures the default value is False
        "help": "Write out CSVs to data/ on the local machine?"
    },
    "index_start": {
        "default": 0,
        "type": int,
        "help": "Array indexing (0-based or 1-based)"
    },
    "data_src": {
        "type": str,
        "choices": ["itv", "ana", "usgs", "usgsi"],
        "help": "For which data source should this script run?"
    },
    "day_tolerance": {
        "default": 8,
        "type": int,
        "help": "Days of search around sample date for a matching Sentinel \
            2 image."
    },
    "cloud_thr": {
        "default": 80,
        "type": int,
        "help": "Percent of cloud cover acceptable in the Sentinel tile \
            corresponding to the sample. If this threshold is surpassed, \
            no Sentinel image chip will be collected for the sample."
    },
    "buffer_distance": {
        "default": 500,
        "type": int,
        "help": "Square search radius (in meters) to use for reflectance data \
            aggregation. This determines the size of the image chip that will \
            be extracted and processed."
    },
    "write_chips": {
        "action": "store_true", # ensures the default value is False
        "help": "Write image chips to blob storage? Should be set to true \
            unless troubleshooting."
    },
    "mask_method1": {
        "default": "lulc",
        "type": str,
        "choices": ["lulc", "scl"],
        "help": "Which data to use for masking (removing) non-water in order \
            to calculate aggreated reflectance values for only water pixels? \
            Choose (\"scl\") to water pixels as identified based on the \
            Scene Classification Layer that accompanies the Snetinel tile, \
            or (\"lulc\") to use Impact Observatory's Land-Use/Land-Cover \
            layer to identify water, and the Scene Classification Layer to \
            remove clouds. Using \"lulc\" is strongly recommended."
    },
    "mask_method2": {
        "default": "mndwi",
        "type": str,
        "choices": ["mndwi", "ndvi", "\"\""],
        "help": "Which additional normalized index to use, if any, to update \
        the mask to remove errors of ommission (pixels classified as water \
        that shouldn't be) prior to calculated aggregated reflectance? If \
        \"ndvi\", then only pixels with an NDVI value less than 0.25 \
        will be retained. If \"mndwi\" (recommended) then only pixels with an \
        MNDWI value greater than 0 will be retained. Of \"\", then no \
        secondary mask is used."
    },
    "n_folds": {
        "default": 5,
        "type": int,
        "help": "The number of folds to create for the training / validation \
            set when fitting models using k-fold cross-validation."
    },
    "seed": {
        "default": 123,
        "type": int,
        "help": "The seed (an integer) used to initialize the pseudorandom \
            number generator for use in partitioning data."
    },
    "rgb_min": {
        "default": 100,
        "type": int,
        "help": "Minimum reflectance value (corresponding to zero saturation \
            of R, G, and B). Used for fine-tuning the visual representation \
            of the chip both for app display and QA/QC chip creation."
    },
    "rgb_max": {
        "default": 4000,
        "type": int,
        "help": "Maximum reflectance value (corresponding to full saturation \
            of R, G, and B). Used for fine-tuning the visual representation \
            of the chip both for app display and QA/QC chip creation."
    },
    "gamma": {
        "default": 0.7,
        "type": float,
        "help": "Gamma correction to use when generating visual image chips. \
            Used for fine-tuning the visual representation \
            of the chip both for app display and QA/QC chip creation."
    },
    "composite": {
        "default": "rgb",
        "type": str,
        "choices": ["rgb", "cir", "swir"],
        "help": "Which color composite to download (for the images used for \
            performing QA). \"rgb\", color infrared (\"cir\"), or short-wave \
            infrared (\"swir\")"
    },
    "n_workers": {
        "default": np.NaN,
        "type": int,
        "help": "How many workers to use for fitting models in parallel \
            (recommended not to go over number of physical cores). If \"nan\", \
            the default, the the number of workers will be set to the number \
            of physical cores (recommended)."
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
    }
}