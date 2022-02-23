import os, pandas as pd, argparse, fsspec

from pandas.core.algorithms import diff

if __name__ == "__main__":
    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_chip_list_name',
        default="good_chips.csv",
        type=str,
        help="The file name for the list of good chips")
    parser.add_argument('--day-tolerance',
        default=8,
        type=int,
        help="accetable deviance (in days) around sample date for USGSI, ITV, and ANA sites")
    parser.add_argument('--cloud-thr',
        default=80,
        type=int,
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer-distance',
        default=500,
        type=int,
        help="search radius used for reflectance data aggregation")
    parser.add_argument('--mask-method1',
        default="lulc",
        choices=["lulc", "scl"],
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask-method2',
        default="mndwi",
        choices=["ndvi", "mndwi", ""],
        type=str,
        help="Which additional index to use, if any, to update the mask, (\"ndvi\") or (\"mndwi\"), or \"\" to use no second mask")

    args = parser.parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    chip_dir = f"data/qa_chips/{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking"
    chip_list_name = f"az://modeling-data/good-chip-lists/{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{args.qa_chip_list_name}"
    
    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}

    fs = fsspec.filesystem("az", **storage_options)

    good_chips = pd.DataFrame({
        "chips": os.listdir(chip_dir)
    })

    good_chips.to_csv(chip_list_name, storage_options=storage_options)