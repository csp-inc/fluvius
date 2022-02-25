10-compile-grid-search-results.py
=================================

This script compiles the grid search results from 09-LMP-grid-search.py by 
parsing and combining the loss stastistics and hyperparameters for each model
into a single CSV, and saving it to Azure Blob Storage. This CSV is then queried 
in 11-fit-top-model.py to identify the best model and refit it using all of the
data.

.. argparse::
   :filename: ../bin/10-compile-grid-search-results.py
   :func: return_parser
   :prog: 10-compile-grid-search-results.py