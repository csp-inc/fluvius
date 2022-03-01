03-image-join.py
================

This script queries the STAC API (the catalog of all remote sensing URLs that 
exist on Azure) for field based observations of suspended sediment 
concentration, and returns a list of image URLs and aggregated features
that match the observations with respect to space and time. This script prepares
inputs and writes them to Azure Blob Storage for use in bin/04-data-merge.py
Optionally, images can be written to cloud storage for use in subsequent QA/QC 
and modeling steps.

.. argparse::
   :filename: ../bin/03-image-join.py
   :func: return_parser
   :prog: 03-image-join.py