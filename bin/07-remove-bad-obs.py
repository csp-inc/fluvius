import os, pandas as pd, argparse
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
