01-usgs-station-acquire.py
==========================

This script runs a web scraper to collect USGS water station data from the National Real-Time Water Quality web tool (https://nrtwq.usgs.gov). The script will query through the full list of available sites and create CSV files for all USGS water quality stations that can be stored to Azure Blob Storage for use in the bin/04-data-merge.py script. 

.. argparse::
   :filename: ../bin/01-usgs-station-acquire.py
   :func: return_parser
   :prog: 01-usgs-station-acquire.py
