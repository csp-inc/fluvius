03-image-join.py
================

This script queries the STAC API (the catalog of all remote sensing URLs that exist on Azure) for field based observations of suspended sediment concentration, and returns a list of image URLs that match the observations with respect to space and time.

.. argparse::
   :filename: ../bin/03-image-join.py
   :func: return_parser
   :prog: 03-image-join.py