11-fit-top-model.py
===================

This script identifies the best model from the grid search and refits it using
all of the data. Model checkpoints and metadata are saved to cloud storage for 
use in subsequent scripts.

.. argparse::
   :filename: ../bin/11-fit-top-model.py
   :func: return_parser
   :prog: 11-fit-top-model.py