from cmath import nan
import pandas as pd
import numpy as np
import os, sys
sys.path.append("/content")
# from src.utils import train_test_validate_split
import datetime as dt
import argparse

def return_parser():
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
    parser.add_argument('--out_filetype',
        default="csv",
        type=str,
        help="filetype for saved merged dataframe (csv or json)")
    parser.add_argument('--mask_method1',
        default="lulc",
        choices=["lulc", "scl"],
        type=str,
        help="Which data to use for masking non-water, scl only (\"scl\"), or io_lulc plus scl (\"lulc\")")
    parser.add_argument('--mask_method2',
        default="mndwi",
        choices=["ndvi", "mndwi", ""],
        type=str,
        help="Which additional index to use to update the mask, (\"ndvi\") or (\"mndwi\"), or \"\" to use no second mask")
    parser.add_argument('--n_folds',
        default=5,
        type=int,
        help="The number of folds to create for the training / validation set")
    parser.add_argument('--seed',
        default=123,
        type=int,
        help="The seed (an integer) used to initialize the pseudorandom number generator")
    return parser

if __name__ == "__main__":
    
    args = return_parser().parse_args()

    # arguments
    chip_size = args.buffer_distance
    day_tolerance = args.day_tolerance
    cloud_thr = args.cloud_thr
    out_filetype = args.out_filetype
    mm1 = args.mask_method1
    mm2 = args.mask_method2
    n_folds = args.n_folds
    seed = args.seed
    rng = np.random.default_rng(seed) # initializes a random number generator

    # hard-coded parameters that influence the final parition, but which are not exposed as args...
    min_water_pixels = 20
    partition_props = {
        "data_src": ["usgs", "ana", "itv"], 
        "p_train": [1, 0.8, 0.1], # [1, 0.9, 0]
        "p_test": [0, 0.2, 0.9] # [0, 0.1, 1]
        }

    # Set storage options for Azure blob storage
    with open("credentials") as f:
        env_vars = f.read().split("\n")

    for var in env_vars:
        key, value = var.split(' = ')
        os.environ[key] = value

    storage_options = {'account_name':os.environ['ACCOUNT_NAME'],
                    'account_key':os.environ['BLOB_KEY']}

    try:
        # filepath = f"data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv"
        # data = pd.read_csv(filepath)
        filepath = f"az://modeling-data/fluvius_data_post_qa_unpartitioned_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking.csv"
        data = pd.read_csv(filepath, storage_options=storage_options)
    except:
        print(f"Error: no file at {filepath}")
    
    ## Add variables for stratifying data partition
    # SSC Quartile
    ssc = np.array(data["SSC (mg/L)"])
    ssc_quantiles = np.quantile(ssc, [0, 0.25, 0.5, 0.75])
    ssc_quantile_bin = np.digitize(ssc, ssc_quantiles)

    # year
    year = [dt.date.fromisoformat(i).year for i in list(data["Date-Time"])]

    # day of year
    julian = [dt.date.fromisoformat(i).timetuple().tm_yday for i in list(data["Date-Time"])]

    # add columns back to data
    data["julian"] = julian
    data["SSC Quantile"] = ssc_quantile_bin
    data["Year"] = year
    
    data["Season"] = np.digitize(np.array(data["julian"]), 366/2 * np.array([0, 1]))
    data["sine_julian"] = np.sin(2*np.pi*data["julian"]/365)
    data["is_brazil"] = 0
    data.loc[data["data_src"].isin(["itv", "ana"]), "is_brazil"] = 1

    # collapse observations designated as 'usgsi' to the overarching 'usgs' source
    data["data_src_raw"] = data["data_src"]
    data["data_src"] = ["usgs" if x == "usgsi" else x for x in data["data_src_raw"]]
    # print(data.groupby('data_src')['data_src'].count())
    # print(data.groupby('data_src_raw')['data_src_raw'].count())

    # filter to remove records with less than the required minimum number of water pixels
    data = data[data["n_water_pixels"] >= min_water_pixels]
    # remove duplicated records
    data.drop_duplicates(["data_src", "site_no", "Date"], inplace = True)

    # develop first partition: assign sites to a train (training + validation) or test set
    # each site appears in one or the other (train or test), but not both
    partition_by = ["data_src", "site_no", "is_brazil"]
    partition_info = data[partition_by].drop_duplicates().set_index("data_src") \
        .join(pd.DataFrame(partition_props).set_index("data_src"), on = "data_src")

    # https://stackoverflow.com/questions/67504101/applying-numpy-random-choice-to-randomise-categories-with-probabilities-from-pan
    # https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    def randomiser(x, rng):
        return rng.choice(["train", "test"], size=(1, 1), p=[x['p_train'], x['p_test']])[0][0]
    partition_info["partition"] = partition_info.apply(lambda x: randomiser(x, rng), axis=1)
    partition_info = partition_info.drop(["p_train", "p_test"], axis = 1).reset_index()

    data_partitioned = data.set_index(partition_by) \
        .join(partition_info.set_index(partition_by), on = partition_by).reset_index()
    partitions_summary = data_partitioned \
        .groupby(["data_src", "is_brazil", "partition"]) \
        .apply(lambda x: pd.Series({
            "n_sites": x["site_no"].nunique(),
            "n_obs": x["sample_id"].nunique()
        })) # does not include explicit zeros (e.g., the zero count for the usgs test partition)
    # print(partitions_summary.reset_index())
    # ps_path = f"data/partitions_summary_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv"
    ps_filepath = f"az://modeling-data/partitions_summary_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.csv"
    # partitions_summary.to_csv(ps_filepath)
    partitions_summary.to_csv(ps_filepath, storage_options=storage_options)

    # create folds
    validation_info = pd.DataFrame()
    training_info = pd.DataFrame()
    for is_brazil in range(2):
        validation_is_brazil = partition_info[(partition_info["partition"] == "train") \
            & (partition_info["is_brazil"] == is_brazil)].copy()
        n_validation = len(validation_is_brazil.index)
        validation_indices = rng.choice(n_validation, size = n_validation, replace = False)
        validation_is_brazil["partition"] = "validate"
        validation_is_brazil["fold"] = \
            pd.cut(validation_indices, bins = n_folds, labels = range(n_folds))
        validation_info = pd.concat([validation_info, validation_is_brazil], axis=0)
        for fold in range(n_folds):
            site_no_in_fold = validation_is_brazil["site_no"][validation_is_brazil["fold"] == fold]
            training_is_brazil = validation_is_brazil[~validation_is_brazil["site_no"].isin(site_no_in_fold)].copy()
            training_is_brazil[["partition", "fold"]] = ["train", fold]
            training_info = pd.concat([training_info, training_is_brazil], axis=0) 
    cv_info = pd.concat([training_info, validation_info], axis=0)
    # print(validation_info.groupby("fold")["fold"].count())

    training_data = data_partitioned[data_partitioned["partition"] == "train"].drop(["partition"], axis = 1)
    data_all_parts = cv_info.set_index(partition_by) \
        .join(training_data.set_index(partition_by), on = partition_by).reset_index()

    validation_data = data_all_parts[data_all_parts["partition"] == "validate"]
    test_data = data_partitioned[data_partitioned["partition"] == "test"].copy()
    test_data["fold"] = nan
    partition_by.extend(["sample_id", "fold"])
    lookup = pd.concat([ \
        validation_data[partition_by].drop_duplicates(), \
        test_data[partition_by].drop_duplicates()], axis=0)

    lookup_indices = ["data_src", "site_no", "is_brazil", "sample_id"]
    out = data.set_index(lookup_indices) \
        .join(lookup.set_index(lookup_indices), on = lookup_indices).reset_index()
    out["fold_idx"] = [x + 1 for x in out["fold"]]
    # print(out.keys())
    # out.drop(["fold"], axis = 1, inplace=True)
    out["partition"] = ["testing" if np.isnan(x) else "training" for x in out["fold"]]

    out_filepath = f"az://modeling-data/partitioned_feature_data_buffer{chip_size}m_daytol8_cloudthr{cloud_thr}percent_{mm1}{mm2}_masking_{n_folds}folds_seed{seed}.{out_filetype}"
    if out_filetype == "csv":
        # out.to_csv(out_filepath)
        out.to_csv(out_filepath, storage_options=storage_options)
    elif out_filetype == "json":
        # "Okay"
        out.to_json(out_filepath, storage_options=storage_options)

    print(f"Done. Outputs written to {out_filepath}")
