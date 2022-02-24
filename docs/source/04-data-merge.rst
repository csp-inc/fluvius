04-data-merge.py
================

This script merges the feature data and URLs obtained for each data source
in bin/03-image-join.py into a single dataframe. This dataframe contains the raw
data that will ultimately be filtered, partitioned, and wrangled in subsequent
workflow steps.

.. argparse::
   :filename: ../bin/04-data-merge.py
   :func: return_parser
   :prog: 04-data-merge.py