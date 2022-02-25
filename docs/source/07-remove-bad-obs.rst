07-remove-bad-obs.py
====================

This script uses the list of high quality samples saved to Azure Blob Storage
in bin/06b-upload-good-chips-list.py to filter the training data.

.. argparse::
   :filename: ../bin/07-remove-bad-obs.py
   :func: return_parser
   :prog: 07-remove-bad-obs.py