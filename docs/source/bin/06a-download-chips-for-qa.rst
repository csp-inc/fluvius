06a-download-chips-for-qa.py
============================

This script downloads the chips generated in bin/05-prep-qa-chip-dataset.py to 
the local filesytem to be used for sample QA/QC. PNGs depicting the sentinel 2 
image and water mask corresponing to the associated training sample will be 
written to a new directory:
data/qa_chips/<buffer-distance>m_cloudthr<cloud_thr>_<mask-method1><mask_method2>_masking/.
One this is run, the data scientist can go through the saved PNG files 
and delete the images that correspond to poor-quality samples. Each PNG is 
named according to the unique ID that corresponds to the sample that it
represents, so in subsequent steps, training samples that should be kept (i.e.
are high-quality) and used in model training can be identified based on which 
files remain on the local filesystem. It is recommended that this script be run
on a local machine as opposed to an Azure VM, as it is easier to preview and 
delete files on a local machine.

.. argparse::
   :filename: ../bin/06a-download-chips-for-qa.py
   :func: return_parser
   :prog: 06a-download-chips-for-qa.py