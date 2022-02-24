09-MLP-grid-search.py
=====================

This script performs the grid search for hyperparameter optimization and saves
loss statistics and other model information from each model that will be used to
identify the "best" model. Individual model results will be saved on the local 
filesystem as JSON files in a folder named according to the arguments to this 
script: 
output/mlp/<buffer-distance>m_cloudthr<cloud-thr>_<mask-method1><mask-method2>_masking_<n-folds>folds_seed<seed>/.
File names for each model output are shortened hashes based on the 
hyperparameters for that given model.

.. argparse::
   :filename: ../bin/09-MLP-grid-search.py
   :func: return_parser
   :prog: 09-MLP-grid-search.py