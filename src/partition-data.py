import pandas as pd
import os

# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.
# pc.settings.set_subscription_key(<YOUR API Key>)
with open(".env") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(' = ')
    os.environ[key] = value

storage_options={'account_name':os.environ['ACCOUNT_NAME'],
                 'account_key':os.environ['BLOB_KEY']}

data = pd.read_json('az://modeling-data/fluvius_data.json',
                    storage_options=storage_options)
