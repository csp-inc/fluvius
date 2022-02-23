import os, pandas as pd, argparse

from pandas.core.algorithms import diff

if __name__ == "__main__":
    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_chip_list_name',
        default="good_chips.csv",
        type=str,
        help="The Azure filename for the list of good chips. " +
             "Assumed to be in the \"modeling-data/chips/post-qa/{chip_size}" +
             "m_cloudthr{cloud_thr}_{mm1}{mm2}_masking` directory\". " +
             "This will be used to determine which samples in the feature " +
             "dataframe to keep.")
    parser.add_argument('--day-tolerance',
        default=8,
        type=int,
        help="accetable deviance (in days) around sample date for USGSI, ITV, and ANA sites")
    parser.add_argument('--cloud_tr',
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
        help="Which additional index, if any, to use to update the mask, (\"ndvi\") or (\"mndwi\"), or \"\" to use no second mask")
    args = parser.parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    chip_list_fn = f"az://modeling-data/good-chip-lists/{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{args.qa_chip_list_name}"

    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}

    all_data = pd.read_csv(f"az://modeling-data/fluvius_data_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv", storage_options=storage_options)

    good_chips = list(pd.read_csv(chip_list_fn, storage_options=storage_options).chips)
    all_data_chip_basename = [x.split("/")[-1] for x in all_data.rgb_and_water_png_href]
    
    keep = [x in good_chips for x in all_data_chip_basename]

    filtered_data = all_data.loc[keep, ]

    filtered_data.to_csv(
        f"az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv",
        storage_options=storage_options
    )

    print("Done!\n" +
        f"output written to az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv"
    )
