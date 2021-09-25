from PIL import Image
import pandas as pd
import rasterio as rio
import os
import numpy as np
import copy
import fsspec
import datetime as dt
import json

with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

chip_size = 500
cloud_thr = 80
write_chips_blob = True
write_chips_local = True
mask_method = "lulc"
rgb_min = 100
rgb_max = 4000
gamma = 0.7

chip_metadata = pd.DataFrame(columns=["region", "site_no", "sample_id", "Date-Time", "rgb_and_water_png_href"])
fs = fsspec.filesystem("az", **storage_options)

n_chip = 0 # initialize
with rio.Env(
        AZURE_STORAGE_ACCOUNT=os.environ["ACCOUNT_NAME"],
        AZURE_STORAGE_ACCESS_KEY=os.environ["BLOB_KEY"]
    ):
    for data_src in ["itv", "ana", "usgsi", "usgs"]:
        chip_paths = fs.ls(f"modeling-data/chips/{chip_size}m_cloudthr{cloud_thr}_{mask_method}_masking/{data_src}")
        chip_obs = [x for x in chip_paths if "water" not in x]
        n_chip += len(chip_obs)
        for path in chip_obs:
            if write_chips_local or write_chips_blob:
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

            out_name = f"app/img/rgb_and_{mask_method}water/{data_src}_{os.path.basename(path[:-4])}.png"

            if write_chips_local:
                qa_img.save(f"data/app/img/rgb_and_{mask_method}water/{data_src}_{os.path.basename(path[:-4])}.png")
            
            if write_chips_blob:
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
    f"az://modeling-data/merged_feature_data_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mask_method}_masking.csv",
    storage_options=storage_options
).assign(region=lambda x: x.data_src).assign(site_no=lambda x: [sp.split("_")[0] for sp in x.sample_id]).reset_index()

all_data = pd.merge(model_data, chip_metadata, how="left", on=["region", "sample_id", "Date-Time"]).dropna().reset_index()

# sites_with_features = model_data[["sample_id", "region"]]
# sites_with_chips = chip_metadata[["sample_id", "region"]]

# features_without_chips = [sites_with_features.iloc[[i]] for i in range(0, sites_with_features.shape[0]-1) if sites_with_features.iloc[i, 0] not in list(sites_with_chips["sample_id"])]
# chips_without_features = [sites_with_chips.iloc[i, :] for i in range(0, sites_with_chips.shape[0]-1) if sites_with_chips.iloc[i, 0] not in list(sites_with_features["sample_id"])]

# features_without_chips = pd.concat(features_without_chips, axis=0)
# chips_without_features = pd.concat(chips_without_features, axis=0)

# missing = [idx for idx in range(0,all_data.shape[0]-1) if type(all_data["water_chip_href"].iloc[idx]) == float]
# missing_from_chip = all_data.iloc[missing,].sample_id

site_no = all_data["site_no_x"]
all_data.drop(["site_no_y", "site_no_x", "level_0", "Unnamed: 0"], axis=1, inplace=True)
all_data.insert(2, "site_no", site_no)

all_data.to_csv(f"az://modeling-data/fluvius_data_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mask_method}_masking.csv.json", storage_options=storage_options)

### Prep JSON for app
site_metadata = pd.read_csv("az://app/station_metadata.csv", storage_options=storage_options)

sites = all_data.site_no.unique()
out_dicts = []
for site in sites:
    site_df = all_data[all_data["site_no"] == site].reset_index()
    region = site_df.region.iloc[0]
    lat = site_df.Latitude.iloc[0]
    lon = site_df.Longitude.iloc[0]

    samples = []
    for i, row in site_df.iterrows():
        samples.append({
            "sample_id": row["sample_id"],
            "SSC.mg.L": str(row["SSC (mg/L)"]),
            "Q.m3.s": str(row["Q (m3/s)"]),
            "sample_date": row["Date-Time"],
            # "sample_julian": str(dt.date.fromisoformat(row["Date-Time"]).timetuple().tm_yday),
            # "acquisition_date": row["Date-Time_Remote"],
            # "acquisition_julian": str(dt.date.fromisoformat(row["Date-Time_Remote"]).timetuple().tm_yday),
            "sentinel.2.l2a_R": str(round(row["sentinel-2-l2a_B04"])),
            "sentinel.2.l2a_G": str(round(row["sentinel-2-l2a_B03"])),
            "sentinel.2.l2a_B": str(round(row["sentinel-2-l2a_B02"])),
            "sentinel.2.l2a_NIR": str(round(row["sentinel-2-l2a_B08"])),
            "Chip.Cloud.Pct": str(round(row["Chip Cloud Pct"])),
            "rgb_water_chip_href": row["rgb_and_water_png_href"]#,
            #"raw_img_chip": row["raw_img_chip_href"]
        })
    
    out_dicts.append({
        "region": row["region"],
        "site_no": site,
        "site_name": [x["site_name"] for i,x in site_metadata.iterrows() if x["site_no"].zfill(8) == site][0],
        "Longitude": lon,
        "Latitude": lat,
        "sample_data": samples
    })


with fs.open('app/all_data_v2.json', 'w') as fn:
    json.dump(out_dicts, fn)
