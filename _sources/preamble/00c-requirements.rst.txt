Requirements
============

Resource Group
**************

Users must create a resource group within the Active Directory they are working in. See this documentation `here <https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal>`__. 

Azure Virtual Machine 
*********************

For individual projects, users can `create a virtual machine <https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal>`_. To allow use of the various open source libraries required for this project, we suggest using an Ubunut Linux based VM. See pricing `here <https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/>`__.

Azure Blob Storage 
******************

Users must have an Azure Blob Storage account and an associated `storage container <https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction>`_ for data to be transfered from the Microsoft Planetary Computer. See following `link <https://planetarycomputer.microsoft.com/>`_ to acquire a planetary computer account.

Docker
******

This pipeline requires Docker to function properly. Please install `Docker Engine <https://docs.docker.com/engine/install/ubuntu/>`_ for your environment, on the same machine to which you cloned this repository.

Authentication
**************

This pipeline requires authentication to both Azure Blob Storage and Planetary Computer. If you do not wish to use your account, and instead use the CSP service account to log into these services, please contact `CSP <https://www.csp-inc.org/about-us/contact-us/>`_. If, however, you wish to use your own credentials for authenticating to Blob Storage and Planetary Computer, please follow the instructions contained within the "Building Environment Variables" section of the User Guide.

Building Credentials
********************

To do store the initial data processing results you will need to provide your credentials by modifying a `credentials` file, which will populate the environment variables used in the script pipeline to send data to blob storage.

Running Docker
**************

The Docker container that runs the pipeline has all the dependencies pre-installed to work for data extraction and modeling. `Example docker run <https://docs.docker.com/engine/reference/run/>`_.
