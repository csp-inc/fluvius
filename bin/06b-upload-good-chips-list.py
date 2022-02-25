import os, pandas as pd, argparse, fsspec

from pandas.core.algorithms import diff
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
    return parser

if __name__ == "__main__":
    
    args = return_parser().parse_args()

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