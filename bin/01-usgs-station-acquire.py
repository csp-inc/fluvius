"""Creates csv files for all the USGS water quality stations.
ISSUES:
    Couple problem stations in the continuous data setting that needs to be resolved (duplicate columnames) 
    Slow and should include concurrent.futures to parallelize the query

Example:
python 01-usgs-station-acquire.py \
    --get-instantaneous=True \
    --write-to-csv=True
"""
import sys
sys.path.append('/content')
from src.fluvius import USGS_Water_DB, USGS_Station
from src.defaults import args_info
from concurrent.futures import ProcessPoolExecutor
import argparse

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--get-instantaneous',
            action=args_info["get_instantaneous"]["action"],
            help=args_info["get_instantaneous"]["help"])
    parser.add_argument('--write-to-csv',
            action=args_info["write_to_csv"]["action"],
            help=args_info["write_to_csv"]["help"])
    parser.add_argument('--index-start',\
            default=args_info["index_start"]["default"],
            type=args_info["index_start"]["type"],
            help=args_info["index_start"]["help"])
    return parser

if __name__ == "__main__":
    args = return_parser().parse_args()

    def collect_write_data(site_no):
        #this function is defined by the global arguments
        water_data = USGS_Station(site_no, instantaneous=args.get_instantaneous) 
        water_data.get_water_df(write_to_csv=args.write_to_csv)

    db = USGS_Water_DB()
    db.get_station_df()
    site_no_list = db.station_df.iloc[args.index_start:].site_no

    with ProcessPoolExecutor() as pool:
        pool.map(collect_write_data, site_no_list)
