06b-upload-good-chips-list.py
=============================

This script identifies the files that remain on the local filesytem after 
performing QA/QC following bin/06a-download-chips-for-qa.py, records their 
unique sample IDs, and uploads this list to Azure Blob Storage to be used later 
on to filter the training data to only include high-quality samples that were
not removed during QA/QC.

.. argparse::
   :filename: ../bin/06b-upload-good-chips-list.py
   :func: return_parser
   :prog: 06b-upload-good-chips-list.py