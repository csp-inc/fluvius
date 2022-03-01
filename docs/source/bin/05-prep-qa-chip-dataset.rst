05-prep-qa-chip-dataset.py
==========================

This script is used to prepare image chips that can be used by the data 
scientist to perform QA/QC and remove poor-quality observations from the 
training data in later steps. Assumes and requires that :code:`--write-chips` 
was supplied as a flag when running bin/03-image-join.py.

.. argparse::
   :filename: ../bin/05-prep-qa-chip-dataset.py
   :func: return_parser
   :prog: 05-prep-qa-chip-dataset.py