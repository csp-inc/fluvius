from PIL import Image
import pandas as pd
import rasterio as rio
import os
import numpy as np
import copy
import fsspec
import argparse
from src.defaults import args_info

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

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
    parser.add_argument('--out-filetype',
        default=args_info["out_filetype"]["default"],
        type=args_info["out_filetype"]["type"],
        choices=args_info["out_filetype"]["choices"],
        help=args_info["out_filetype"]["help"])
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
    parser.add_argument('--write-qa-chips',
        action=args_info["write_qa_chips"]["action"],
        help=args_info["write_qa_chips"]["help"])
    parser.add_argument('--rgb-min',
        default=args_info["rgb_min"]["default"],
        type=args_info["rgb_min"]["type"],
        help=args_info["rgb_min"]["help"])
    parser.add_argument('--rgb-max',
        default=args_info["rgb_max"]["default"],
        type=args_info["rgb_max"]["type"],
        help=args_info["rgb_max"]["help"])
    parser.add_argument('--gamma',
        default=args_info["gamma"]["default"],
        type=args_info["gamma"]["type"],
        help=args_info["gamma"]["help"])
    parser.add_argument('--local-outpath',
        default=args_info["local_outpath"]["default"],
        type=args_info["local_outpath"]["type"],
        help=args_info["local_outpath"]["help"])
    return parser

if __name__ == "__main__":

    args = return_parser().parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    write_qa_chips = args.write_qa_chips

    mm1 = args.mask_method1
    mm2 = args.mask_method2
    rgb_min = args.rgb_min
    rgb_max = args.rgb_max
    gamma = args.gamma

    composites = ['rgb', 'cir', 'swir']
    band_combos = { # https://gisgeography.com/sentinel-2-bands-combinations/
        # natural ('true') color
        'rgb': (4, 3, 2), # Earth as humans would see it naturally
        # false color
        'cir': (5, 4, 3), # emphasizes vegetation health (clear water is black, muddy waters look blue)
        'swir': (11, 5, 3) # water is black, sediment-laden water and saturated soil will appear blue
    }

    chip_metadata = pd.DataFrame(columns=["region", "site_no", "sample_id", "Date-Time"] + [f"{x}_and_water_png_href" for x in composites])
    fs = fsspec.filesystem("az", **storage_options)

    n_chip = 0 # initialize
    with rio.Env(
            AZURE_STORAGE_ACCOUNT=os.environ["ACCOUNT_NAME"],
            AZURE_STORAGE_ACCESS_KEY=os.environ["BLOB_KEY"]
        ):
        for data_src in ["itv", "ana", "usgsi", "usgs"]:
            chip_paths = fs.ls(f"modeling-data/chips/{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_src}")
            chip_obs = [x for x in chip_paths if "water" not in x]
            n_chip += len(chip_obs)
            for path in chip_obs:
                for composite in composites:
                    if (args.local_outpath != None) or write_qa_chips:
                        with rio.open(f"az://{path}") as chip:
                            rgb_raw = np.moveaxis(chip.read(band_combos[composite]), 0, -1)
                            with rio.open(f"az://{path[:-4]}_water.tif") as mask:
                                water = mask.read(1)
                        rgb = (
                            ((np.clip(rgb_raw, rgb_min, rgb_max) - rgb_min) /
                                (rgb_max - rgb_min)) ** gamma * 255
                            ).astype(np.uint8)
                        water_rgb = copy.deepcopy(rgb)

                        # Set NA pixels to stand-out color based on 
                        if composite == "rgb":
                            water_rgb[water==0, 0] = 252
                            water_rgb[water==0, 1] = 184
                            water_rgb[water==0, 2] = 255
                        elif composite == "cir":
                            water_rgb[water==0, 0] = 166
                            water_rgb[water==0, 1] = 232
                            water_rgb[water==0, 2] = 139
                        elif composite == "swir":
                            water_rgb[water==0, 0] = 252
                            water_rgb[water==0, 1] = 184
                            water_rgb[water==0, 2] = 255

                        qa_array = np.concatenate([rgb, water_rgb], axis = (1))
                        qa_img = Image.fromarray(qa_array, "RGB")

                    out_name = f"modeling-data/chips/qa/{composite}_{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_src}_{os.path.basename(path[:-4])}.png"

                    if args.local_outpath is not None:
                        if not os.path.exists(f'{args.local_outpath}'):
                            os.makedirs(f'{args.local_outpath}')
                        qa_img.save(f"{args.local_outpath}/{composite}_{data_src}_{os.path.basename(path[:-4])}.png")

                    if write_qa_chips:
                        with fs.open(out_name, "wb") as fn:
                            qa_img.save(fn, "PNG")

                raw_img_url = f"https://fluviusdata.blob.core.windows.net/{path}"
                rgb_water_url = f"https://fluviusdata.blob.core.windows.net/modeling-data/chips/qa/rgb_{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_src}_{os.path.basename(path[:-4])}.png"
                cir_water_url = f"https://fluviusdata.blob.core.windows.net/modeling-data/chips/qa/cir_{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_src}_{os.path.basename(path[:-4])}.png"
                swir_water_url = f"https://fluviusdata.blob.core.windows.net/modeling-data/chips/qa/swir_{chip_size}m_cloudthr{cloud_thr}_{mm1}{mm2}_masking/{data_src}_{os.path.basename(path[:-4])}.png"

                info = os.path.basename(path[:-4]).split("_")
                chip_metadata = pd.concat(
                    [chip_metadata,
                    pd.DataFrame({
                        "region": [data_src],
                        "site_no": [info[0]],
                        "sample_id": [f"{info[0]}_{info[1]}"],
                        "Date-Time": [info[2]],
                        "raw_img_chip_href": [raw_img_url],
                        "water_chip_href": [f"{raw_img_url[:-4]}_water.tif"],
                        "rgb_and_water_png_href": [rgb_water_url],
                        "cir_and_water_png_href": [cir_water_url],
                        "swir_and_water_png_href": [swir_water_url]
                    })
                    ],
                    ignore_index=True
                )

    ## Merge reflectance and response data to image chip hrefs
    model_data = pd.read_csv(
        f"az://modeling-data/merged_feature_data_buffer{chip_size}m_daytol{day_tol}_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv",
        storage_options=storage_options
    ).assign(region=lambda x: x.data_src).assign(site_no=lambda x: [sp.split("_")[0] for sp in x.sample_id]).reset_index()

    all_data = pd.merge(model_data, chip_metadata, how="left", on=["region", "sample_id", "Date-Time"]).dropna().reset_index()

    site_no = all_data["site_no_x"]
    all_data.drop(["site_no_y", "site_no_x", "level_0", "Unnamed: 0"], axis=1, inplace=True)
    all_data.insert(2, "site_no", site_no)

    all_data.to_csv(f"az://modeling-data/fluvius_data_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv", storage_options=storage_options)
