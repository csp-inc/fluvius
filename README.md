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

### Resource Group

Users must create a resource group within the Active Directory they are working in. See this documentation [here](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal). 

### Azure Virtual Machine 

For individual projects, users can [create a virtual machine](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal). To allow use of the various open source libraries required for this project, we suggest using an Ubunut Linux based VM. [See pricing here](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)

### Azure Blob Storage 

Users must have an Azure Blob Storage account and an associated [storage container](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) for data to be transfered from the Microsoft Planetary Computer. See following [link](https://planetarycomputer.microsoft.com/) to acquire a planetary computer account.

### Docker

This pipeline requires Docker to function properly. Please install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) for your environment, on the same machine to which you cloned this repository.

### Authentication

This pipeline requires authentication to both Azure Blob Storage and Planetary Computer. If you do not wish to use your account, and instead use the CSP service account to log into these services, please contact [CSP](https://www.csp-inc.org/about-us/contact-us/). If, however, you wish to use your own credentials for authenticating to Blob Storage and Planetary Computer, please follow the instructions contained within the "Building Environment Variables" section of the User Guide.

### Building Credentials

To do store the initial data processing results you will need to provide your credentials by modifying a `credentials` file, which will populate the environment variables used in the script pipeline to send data to blob storage.

### Running Docker

The Docker container that runs the pipeline has all the dependencies pre-installed to work for data extraction and modeling. [Example docker run](https://docs.docker.com/engine/reference/run/)

## Quick(ish) Start

Given the technical nature of this cloud environment, we describe here a workflow where a user can connect to an Azure Virtual Machine and use the terminal similar to a local based server. 

### Secure Shell (SSH)

SSH or Secure Shell is a network communication protocol that enables two computers to communicate (c.f http or hypertext transfer protocol, which is the protocol used to transfer hypertext such as web pages) and share data. Depending on the user's local machine operating system, there are various means to [connect via SSH into a running Azure VM](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/mac-create-ssh-keys). Below we describe connecting through a Unix based OS (Linux or MacOSX). 

#### SSH key gen

First one needs to create a key pair for the local machine to connect the VM. This can be done by opening a terminal and typing:

```
ssh-keygen -t rsa -b 4096 -f key-filename
```
![keygif](https://fluviusdata.blob.core.windows.net/example/key_example.gif)
This command will create two files, `key-filename` which is the private key and `key-filename.pub` which is the public key. The public key will be submitted to the VM to connect with the local machine and the `key-filename` file is the private key that will confirm that prrof of the user's identity that should live only on the local machine. 

Public key submission can be provided to the Azure VM through the GUI, seen in the gif below.
![keysubmission](https://fluviusdata.blob.core.windows.net/example/ssh_submission.gif)

Alternatively, one can add their public key and add a user to an existing VM using the `Reset password` blade on the Azure VM portal GUI. Note that this does not actually reset the password unless the user already exists. If you are adding a user that does not exist, this creates a new profile within the VM. This means multiple users can connect via ssh with their own unique key to a specific VM. 

![resetssh](https://fluviusdata.blob.core.windows.net/example/add_user.gif)

#### Connect via SSH

Once the key is generated and public key submitted to the VM, the user needs only start the VM and then connect via ssh from the terminal to the VM. This is now a virtual environment that can be accessed anywhere. 

Go to the terminal and type:

```
ssh -i $PUBLICKEYFILENAME\
    -L localhost:$PORT:$DNS:$PORT \
    $USER@$DNS
```
where the following variables are:

- $PUBLICKEYFILENAME : The public key that you generated with ssh-keygen (e.g key-example.pub)
- $PORT : The port that you wish to mount to the VM (e.g. 8888)
- $DNS : The IP address for the VM or DNS name (e.g. vm-brazilsouth.cloudapp.azure.com)
- $USER : The username associated to ssh-key (e.g. itv-user)

Note that the $PORT must be [added to the VM](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal) and not already allocated by the local machine. This is commonly '8888' or '8080', but can be anything the user wants to share with the VM. This will be useful when connecting to a Jupyter Notebook instantiated by the VM. 

Using a DNS name for $DNS is useful to not require a new dynamic IP address for the VM each time it is turned on. A DNS can be assign following this [guide](https://docs.microsoft.com/en-us/azure/virtual-machines/custom-domain#add-custom-domain-to-vm-public-ip-address). 

## Authors

* **Tony Chang** - *Principle Investigator* - [CSP](http://www.csp-inc.org/about-us/core-science-staff/chang-tony/)

## Contributors

* **Vincent Landau** - *Data Scientist* - [CSP](http://www.csp-inc.org/about-us/core-science-staff/landau-vincent/)



