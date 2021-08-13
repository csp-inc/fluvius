# fluvius
# Welcome to project fluvius

Project fluvius is a collaboration with [Analytics Lab at Conservation Science Partners](https://analytics-lab.org/), the [Instituto Tecnologico Vale](www.itv.org), and [Microsoft Brazil](https://www.microsoft.com/en-us/ai/ai-for-earth) for estimating river health in the Itacaiunas River Basin.

## Background 
Since the 1970s, forests in the Amazon region have been increasingly converted to pasturelands, human settlement areas, and natural resource extraction regions. By 2006, almost 95% of all deforestation in the Brazilian Amazon occurred within 5.5 km of roadways or less than 1 km from navigable rivers with a number of consequences for freshwater ecosystems, including changes in runoff characteristics. One measurable impact on the riparian system is greater discharge and increased sediment flux in the channels of the main rivers which can be used as a proxy for ecosystem degradation (Coe et al., 2011; Costa et al.; 2003; Latrubesse et al.; 2009). Therefore, one means of understanding the impact of deforestation or mitigation through reforestation is by monitoring the sediment flux over time. This understanding of the dynamics of hydro-sedimentological processes in river basins provides critical data for decision-making and supports management planning for the rational use of natural resources. However, due to the logistical difficulties in measuring sediment flux in a distributed way through time and space, there is a significant opportunity to increase monitoring frequency of water bodies through satellite remote sensing. 

New artificial intelligence techniques such as convolutional neural networks (CNN) have shown promising performance in the application of object detection and continuous forest structure estimation when combined with remote sensing imagery (Chang et al. 2019). CNNs allow additional information gained through image structure that can often outperform traditional remote sensing approaches that are limited to spectral analysis alone. This presents an opportunity to 

## What are the goals of project fluvius?

The central focus of project fluvius aims to understand the response of river basins after forest disturbancea and restoration. Several other goals are also noted in the advancement of remote sening, artificial intelligence, and ecosystem monitoring. 

Project goals:

- Co-development of an automated pipeline for core datasets, preprocessing, and storage on a cloud based platform  
- Prototyping of various experimental deep learning models to analyze remote sensed images of rainforest watersheds and estimate the total suspended sediment (TSS)
- Diagnose the deep learning model on field based measurements and compare to performance of current state-of-the-art spectral based models
- Deploy the deep learning model API on the Itacaiúnas River watershed over space and time

## Requirements
### Azure Blob Storage 

Users must have an Azure Blob Storage account and an associated [storage container](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) for data to be transfered from the Microsoft Planetary Computer. See following [link](https://planetarycomputer.microsoft.com/) to acquire a planetary computer account.

### Docker

This pipeline requires Docker to function properly. Please install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) for your environment, on the same machine to which you cloned this repository.

### Authentication

This pipeline requires authentication to both Azure Blob Storage and Planetary Computer. If you do not wish to use your account, and instead use the CSP service account to log into these services, please contact [CSP](https://www.csp-inc.org/about-us/contact-us/). If, however, you wish to use your own credentials for authenticating to Blob Storage and Planetary Computer, please follow the instructions contained within the "Building Environment Variables" section of the User Guide.

### Building Credentials
To do store the initial data processing results you will need to provide your credentials by modifying a `credentials` file, which will populate the environment variables used in the script pipeline to send data to blob storage.

### Running Docker

The Docker container that runs the pipeline has all the dependencies pre-installed to work for data extraction and modeling. 

## Authors

* **Tony Chang** - *Principle Investigator* - [CSP](http://www.csp-inc.org/about-us/core-science-staff/chang-tony/)

## Contributors

* **Vincent Landau** - *Data Scientist* - [CSP](http://www.csp-inc.org/about-us/core-science-staff/landau-vincent/)



