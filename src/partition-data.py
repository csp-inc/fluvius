import pandas as pd
import numpy as np
import os, sys
import datetime as dt

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
with open(".env") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(' = ')
    os.environ[key] = value

storage_options = {'account_name':os.environ['ACCOUNT_NAME'],
                   'account_key':os.environ['BLOB_KEY']}

def train_test_validate_split(df, proportions, part_colname = "partition"):
    """
    Takes a DataFrame (`df`) and splits it into train, test, and validate
    partitions. Returns a DataFrame with a new column, `part_colname` specifying
    which partition each row belongs to. `proportions` is a list of length 3 with 
    desired proportions for train, test, and validate partitions, in that order.
    """
    if sum(proportions) != 1 | len(proportions) != 3:
        sys.exit("Error: proportions must be length 3 and sum to 1.")
    
    # first sample train data
    train = df.sample(frac=proportions[0], random_state=2)
    train[part_colname] = "train"
    # drop train data from the df
    test_validate = df.drop(train.index)
    # sample test data
    test = test_validate.sample(frac=proportions[1]/sum(proportions[1:3]), random_state=2)
    test[part_colname] = "test"
    #drop test data from test_validate, leaving you with validate in correct propotion
    validate = test_validate.drop(test.index)
    validate[part_colname] = "validate"

    return pd.concat([train, test, validate])


data = pd.read_json('az://modeling-data/fluvius_data.json',
                    storage_options=storage_options)

## Add variables for stratifying data partition
# SSC Quartile
ssc = np.array(data["SSC (mg/L)"])
ssc_quantiles = np.quantile(ssc, [0, 0.5])
ssc_quantile_bin = np.digitize(ssc, ssc_quantiles)

# year
year = [dt.date.fromisoformat(i).year for i in list(data["Date-Time"])]

# "season" break into quarters of the year
julian_partition = np.digitize(np.array(data["julian"]), 366/2 * np.array([0, 1]))

# add columns back to data
data["SSC Quantile"] = ssc_quantile_bin
data["Year"] = year
data["Season"] = julian_partition

## Partition the data into train, test, validate
# First split data into groups to ensure stratified
grouped = data.groupby(by = ["SSC Quantile", "Season", "region"], group_keys=False)
# now apply the train_test_validate_split function to each group
partitioned = grouped.apply(lambda x: train_test_validate_split(x, [0.7, 0.15, 0.15]))

partitioned.to_json('az://modeling-data/fluvius_data_partitioned.json',
                    storage_options=storage_options)
