14-make-prediction-chips.py
===========================

This script applies our model to generate predictions at the pixel level for 
each image chip generated from 12-prediction-inputs.py. These prediction results
are intended for visualization purposes only. Our model was designed to make 
predictions based on aggregated reflectance information from multiple pixels, 
not individual pixel, so consumers of this information should not attempt to
draw any inference from the predictions generated by this script. For this 
step, it's only necessary to run the function for ITV and ANA sites, as we are
not deploying predictions for USGS sites to the web app.

.. argparse::
   :filename: ../bin/14-make-prediction-chips.py
   :func: return_parser
   :prog: 14-make-prediction-chips.py