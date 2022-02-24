12-prediction-inputs.py
=======================

This script queries STAC on the Planetary Computer to download and process image
chips from the entire Sentinel 2 catalogue for every ANA and ITV monitoring
site. These images chips will be used subsequently to 
generate time-series model predictions for each site. This script must be run
separately for ANA and ITV sites.

.. argparse::
   :filename: ../bin/12-prediction-inputs.py
   :func: return_parser
   :prog: 12-prediction-inputs.py