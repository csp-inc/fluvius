import fsspec, os, argparse, pandas as pd, shutil

if __name__ == "__main__":
    ############### Parse commnd line args ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--day_tolerance',
        default=8,
        type=int,
        help="accetable deviance (in days) around sample date for USGSI, ITV, and ANA sites")
    parser.add_argument('--cloud_thr',
        default=80,
        type=int,
        help="percent of cloud cover acceptable")
    parser.add_argument('--buffer_distance',
        default=500,
        type=int,
        help="search radius used for reflectance data aggregation")
    parser.add_argument('--mask_method1',
        default="lulc",
        choices=["lulc", "scl"],
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',
        default="mndwi",
        choices=["ndvi", "mndwi", ""],
        type=str,
        help="Which additional index to use, if any, to update the mask, (\"ndvi\") or (\"mndwi\"), or \"\" to use no second mask")
    parser.add_argument("--composite",
        default="rgb",
        type=str,
        help="Which color composite to download. \"rgb\", color infrared (\"cir\"), or short-wave infrared (\"swir\")")
    args = parser.parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    composite = args.composite
    local_save_dir = f"data/qa_chips/{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking"

    with open("/content/credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(" = ")
        os.environ[key] = value

    storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                       "account_key":os.environ["BLOB_KEY"]}

    fs = fsspec.filesystem("az", **storage_options)

    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    else: # start fresh if dir already exists
        shutil.rmtree(local_save_dir)
        os.makedirs(local_save_dir)

    all_data = pd.read_csv(f"az://modeling-data/fluvius_data_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv", storage_options=storage_options)
    remote_path_basenames = [x.split("/")[-1] for x in all_data.rgb_and_water_png_href]

    for path in remote_path_basenames:
        if path in remote_path_basenames:
            fs.get_file(
                f"modeling-data/chips/qa/{composite}_{chip_size}m_" + \
                    f"cloudthr{cloud_thr}_{mm1}{mm2}_masking/{path}",
                f"{local_save_dir}/{path}"
            )
    print(
        f"\nDone! \n\n" +
        f"Now delete the QA chips in {local_save_dir} that represent \n" +
        "bad samples. Then run bin/06b-upload-good-chips.py \n" +
        "to upload the list of good chips to Azure blob storage."
    )
