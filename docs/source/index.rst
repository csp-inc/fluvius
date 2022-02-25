.. fluvius documentation master file, created by
   sphinx-quickstart on Mon Feb 21 16:54:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the fluvius documentation!
=====================================

.. important::

    This documentation was generated on |today|, and is rebuilt on push events 
    to the main branch.

Project Fluvius uses satellite imagery and AI to monitor the health of rivers in
the Amazon and the U.S. by enabling near real-time prediction of suspended
sediment concentration. This website serves as documetation for the code in the
project's `GitHub repository <https://github.com/csp-inc/fluvius>`_. The web app
can be viewed `here <https://fluviuswebapp.z22.web.core.windows.net>`_.

.. raw:: html

   <div class="container">
      <iframe class="responsive-iframe" src="https://earthengine.google.com/iframes/timelapse_player_embed.html#v=-2.26579,-53.55277,5.657,latLng&t=1.45&ps=100&bt=19840101&et=20201231&startDwell=0&endDwell=0" frameborder="0" allowfullscreen></iframe>
   </div>
   <p style="margin-bottom:1cm;"></p>

In the following sections, we provide an introduction to the project, and setup
information.

.. toctree::
   :hidden:
   :caption: Home
   :maxdepth: 0

   self

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   preamble/00a-background
   preamble/00b-project-goals
   preamble/00c-requirements
   preamble/00d-quickstart

The following command line scripts represent the building blocks of the fluvius
workflow. These scripts are meant to be run in the order specified by the
numeric prefix in the filenames. All of these files can be found in 'bin/', and 
should be executed from from the Fluvius repo's root directory (e.g.
:code:`python3 bin/01-usgs-station-acquire.py`)

.. toctree::
   :caption: Command Line Scripts
   :maxdepth: 1

   bin/01-usgs-station-acquire
   bin/02-preprocess-data
   bin/03-image-join
   bin/04-data-merge
   bin/05-prep-qa-chip-dataset
   bin/06a-download-chips-for-qa
   bin/06b-upload-good-chips-list
   bin/07-remove-bad-obs
   bin/08-partition-data
   bin/09-MLP-grid-search
   bin/10-compile-grid-search-results
   bin/11-fit-top-model
   bin/12-prediction-inputs
   bin/13-predict-tabular
   bin/14-make-prediction-chips
   bin/15-prep-data-for-app


Authors
=======
- `Vincent A. Landau <https://github.com/vlandau>`_ - Technical co-lead
- `Luke J. Zachmann <https://github.com/lzachmann>`_ - Technical co-lead
- `Tony Chang <https://github.com/tonychangmsu>`_ - Principal Investigator