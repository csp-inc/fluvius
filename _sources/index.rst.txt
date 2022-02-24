.. fluvius documentation master file, created by
   sphinx-quickstart on Mon Feb 21 16:54:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the fluvius documentation!
=====================================

.. important::

    This documentation was generated on |today|, and is rebuilt on push events to the main branch.

Project Fluvius uses satellite imagery and AI to monitor the health of rivers in the Amazon and the U.S. by enabling near real-time prediction of suspended sediment concentration. The web app is visible `here <https://fluviuswebapp.z22.web.core.windows.net>`_.

In the following sections, we provide an introduction to the project, and setup information.

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   00a-background
   00b-project-goals
   00c-requirements
   00d-quickstart

The following command line scripts represent the building blocks of the fluvius workflow. These scripts are meant to be run in the order specified by the numeric prefix in the filenames. All of these files can be found in 'bin/'.

.. toctree::
   :caption: Command Line Scripts (bin)
   :maxdepth: 1

   01-usgs-station-acquire
   02-preprocess-data
   03-image-join
   04-data-merge
   05-prep-qa-chip-dataset
   06a-download-chips-for-qa
   06b-upload-good-chips-list
   07-remove-bad-obs
   08-partition-data
   09-MLP-grid-search
   10-compile-grid-search-results
   11-fit-top-model
   12-prediction-inputs
   13-predict-tabular
   14-make-prediction-chips
   15-prep-data-for-app
