{mm1}{mm2}import os, pandas as pd, argparse

from pandas.core.algorithms import diff

if __name__ == "__main__":
    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--day_tolerance',\
        default=8,\
        type=int,\
        help="accetable deviance (in days) around sample date for USGSI, ITV, and ANA sites")
    parser.add_argument('--cloud_thr',\
        default=80,\
        type=int,\
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',\
        default=500,\
        type=int,\
        help="search radius used for reflectance data aggregation")
    parser.add_argument('--mask_method1',\
        default="lulc",\
        type=str,\
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',\
        default="",\
        type=str,\
        help="Which additional index to use to update the mask, (\"ndvi\") or (\"mndwi\")")
    parser.add_argument('--qa_chip_dir',\
        default="data/qa_chips",\
        type=str,\
        help="The local directory containing filtered QA chips. The remaining chips will be used to determine which samples in the feature dataframe to keep.")
    args = parser.parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    chip_dir = args.qa_chip_dir

    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}

    all_data = pd.read_csv(f"az://modeling-data/fluvius_data_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv", storage_options=storage_options)

    good_chips = os.listdir(chip_dir)

    all_data_chip_basename = [x.split("/")[-1] for x in all_data.rgb_and_water_png_href]
    set(good_chips).difference(set(all_data_chip_basename))
    keep = [x in good_chips for x in all_data_chip_basename]

    filtered_data = all_data.loc[keep, ]

    filtered_data.to_csv(
        f"az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv",
        storage_options=storage_options
    )

    print("Done!\n" +
        f"output written to az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv"
    )
