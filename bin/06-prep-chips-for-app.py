from PIL import Image
import pandas as pd
import rasterio as rio
import os
import numpy as np
import copy
import fsspec

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

chip_size = 500
cloud_thr = 80
write_chips = False
rgb_min = 100
rgb_max = 4000
gamma = 0.7

chip_metadata = pd.DataFrame(columns=["region", "site_no", "sample_id", "Date-Time", "rgb_and_water_png_href"])
n_chip = 0
with rio.Env(
        AZURE_STORAGE_ACCOUNT=os.environ["ACCOUNT_NAME"],
        AZURE_STORAGE_ACCESS_KEY=os.environ["BLOB_KEY"]
    ):
    for data_src in ["itv", "ana", "usgsi", "usgs"]:
        fs = fsspec.filesystem("az", **storage_options)
        chip_paths = fs.ls(f"modeling-data/chips/{chip_size}m_cloudthr{cloud_thr}/{data_src}")
        chip_obs = [x for x in chip_paths if "water" not in x]
        n_chip += len(chip_obs)
        for path in chip_obs:
            if write_chips:
                with rio.open(f"az://{path}") as chip:
                    rgb_raw = np.moveaxis(chip.read((4, 3, 2)), 0, -1)
                    with rio.open(f"az://{path[:-4]}_water.tif") as mask:
                        water = mask.read(1)
                rgb = (
                    ((np.clip(rgb_raw, rgb_min, rgb_max) - rgb_min) / 
                        (rgb_max - rgb_min)) ** gamma * 255
                    ).astype(np.uint8)
                water_rgb = copy.deepcopy(rgb)
                water_rgb[water==0, :] = 0

                qa_array = np.concatenate([rgb, water_rgb], axis = (1))^2
                qa_img = Image.fromarray(qa_array, "RGB")

            out_name = f"app/img/rgb_and_sclwater/{data_src}_{os.path.basename(path[:-4])}.png"

            if write_chips:
                with fs.open(out_name, "wb") as fn:
                    qa_img.save(fn, "PNG")

            rgb_water_url = f"https://fluviusdata.blob.core.windows.net/{out_name}"
            raw_img_url = f"https://fluviusdata.blob.core.windows.net/{path}"
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
                     "rgb_and_water_png_href": [rgb_water_url]
                 })
                ],
                ignore_index=True
            )

## Merge reflectance and response data to image chip hrefs
model_data = pd.read_csv(
    f"az://modeling-data/merged_training_data_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent.csv",
    storage_options=storage_options
).assign(region=lambda x: x.data_src).assign(site_no=lambda x: [sp.split("_")[0] for sp in x.sample_id]).reset_index()

all_data = pd.merge(model_data, chip_metadata, how="left", on=["region", "sample_id", "Date-Time"]).reset_index().dropna()

sites_with_features = model_data[["sample_id", "region"]]
sites_with_chips = chip_metadata[["sample_id", "region"]]

features_without_chips = [sites_with_features.iloc[[i]] for i in range(0, sites_with_features.shape[0]-1) if sites_with_features.iloc[i, 0] not in list(sites_with_chips["sample_id"])]
chips_without_features = [sites_with_chips.iloc[i, :] for i in range(0, sites_with_chips.shape[0]-1) if sites_with_chips.iloc[i, 0] not in list(sites_with_features["sample_id"])]

features_without_chips = pd.concat(features_without_chips, axis=0)
chips_without_features = pd.concat(chips_without_features, axis=0)

# missing = [idx for idx in range(0,all_data.shape[0]-1) if type(all_data["water_chip_href"].iloc[idx]) == float]
# missing_from_chip = all_data.iloc[missing,].sample_id


## all_data.to_json("az://app/fluvius_data_v2_missing_chips.json", storage_options=storage_options)
