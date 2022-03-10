02-preprocess-data.py
=====================

This script preprocesses the raw water station data based on some hard coded attributes. Note that the intention of this script is for processing USGS, ANA, and ITV data to a standardized format. If using other datasets, it is desirable to modify new data to match those in this study with a column for 'Date-Time'.  

.. argparse::
   :filename: ../bin/02-preprocess-data.py
   :func: return_parser
   :prog: 02-preprocess-data.py
