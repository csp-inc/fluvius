08-partition-data.py
====================

This script develops the partitions used to train and evaluate fluvius models.
A first-phase partition divides the sites into either a training or a test set.
Sites in the test set are never seen by the model during training, including the
hyperparameter grid search used to identify the top-performing model. Sites that
appear in the training set are further partitioned into train and validation
sets over k-folds. Several parameters involved in the partition are
hard-coded.

.. argparse::
   :filename: ../bin/08-partition-data.py
   :func: return_parser
   :prog: 08-partition-data.py