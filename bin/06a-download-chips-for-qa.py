import fsspec, os, argparse, pandas as pd, shutil
from src.defaults import args_info

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day-tolerance',
        default=args_info["day_tolerance"]["default"],
        type=args_info["day_tolerance"]["type"],
        help=args_info["day_tolerance"]["help"])
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
    parser.add_argument("--composite",
        default=args_info["composite"]["default"],
        type=args_info["composite"]["type"],
        choices=args_info["composite"]["choices"],
        help=args_info["composite"]["help"])
    return parser

if __name__ == "__main__":

    args = return_parser().parse_args()

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
