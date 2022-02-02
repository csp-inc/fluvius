from PIL import Image
import pandas as pd
import rasterio as rio
import os
import numpy as np
import copy
import fsspec
import argparse

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

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
    parser.add_argument('--out_filetype',\
        default="csv",\
        type=str,\
        help="filetype for saved merged dataframe (csv or json)")
    parser.add_argument('--mask_method1',\
        default="lulc",\
        type=str,\
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',\
        default="",\
        type=str,\
        help="Which additional index to use, if any, to update the mask, (\"ndvi\") or (\"mndwi\")")
    parser.add_argument('--write_chips_blob',\
        default=True,\
        type=bool,\
        help="Should QA chips be written to blob storage?")
    parser.add_argument('--rgb_min',\
        default=100,\
        type=int,\
        help="Minimum reflectance value (corresponding to zero saturation of R, G, and B)")
    parser.add_argument('--rgb_max',\
        default=4000,\
        type=int,\
        help="Maximum reflectance value (corresponding to full saturation of R, G, and B)")
    parser.add_argument('--gamma',\
        default=0.7,\
        type=float,\
        help="Gamma correction to use when generating the image chip")
    parser.add_argument('--local_outpath',\
        default=None,\
        type=str,\
        help="If desired, the local directory to which QA chips will be saved. If \"None\", the default, chips are not written locally")
    args = parser.parse_args()

    chip_size = args.buffer_distance
    cloud_thr = args.cloud_thr
    day_tol = args.day_tolerance
    write_chips_blob = args.write_chips_blob

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
                    if (args.local_outpath != None) or write_chips_blob:
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

                    if write_chips_blob:
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
