FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y -q install \
    bash \
    build-essential \
    curl \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    nano \
    vim \
    git \
    gcc \
    jags \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python-setuptools \
    tzdata \
    software-properties-common \
    libffi-dev \ 
    libgeos-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libspatialindex-dev \
    fonts-dejavu \
    make \
    gfortran \
    wget \ 
    libpango1.0-0 \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    sudo

#Dotnet install
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb 
RUN sudo dpkg -i packages-microsoft-prod.deb 
RUN apt update && apt install -y apt-transport-https dotnet-sdk-3.1
#RUN wget http://ftp.us.debian.org/debian/pool/main/i/icu/libicu63_63.2-3_amd64.deb
#RUN sudo dpkg -i libicu63_63.2-3_amd64.deb && apt-get install -y dotnet-runtime-2.1 

# install google chrome
#RUN sudo bash -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# Install ChromeDriver.
#RUN CHROME_DRIVER_VERSION=`curl -sS https://chromedriver.storage.googleapis.com/LATEST_RELEASE`
#RUN wget -N https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/chromedriver_linux64.zip -P ~/
#RUN unzip ~/chromedriver_linux64.zip -d ~/
#RUN rm ~/chromedriver_linux64.zip
#RUN mv -f ~/chromedriver /usr/local/bin/chromedriver
#RUN chown root:root /usr/local/bin/chromedriver
#RUN chmod 0755 /usr/local/bin/chromedriver
#RUN ln -s /usr/local/share/chromedriver /usr/local/bin/chromedriver
#RUN ln -s /usr/local/share/chromedriver /usr/bin/chromedriver

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3.8-distutils 
#python 3.8 pip 
RUN curl -o /tmp/get_pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.8 /tmp/get_pip.py
RUN pip3 install \
    azure-cli-core==2.26.1 \
    azure-storage-blob==12.8.1 \
    adlfs==2021.7.1 \
    numpy==1.21.1 \
    pandas==1.3.1 \
    geopandas==0.9.0 \
    descartes==1.1.0 \
    shapely==1.7.1 \
    folium==0.12.1 \
    fsspec==2021.7.0 \
    python-dateutil==2.8.2 \
    simplejson==3.17.4 \
    anyio==3.3.3 \
    rasterio==1.2.6 \
    dask[complete]==2021.7.1 \
    dask-geopandas==0.1.0a4 \
    rtree==0.9.7 \
    xarray==0.19.0 \
    seaborn==0.11.1 \
    jupyterlab==3.0.16 \
    Pillow==8.3.1 \
    matplotlib==3.4.2 \
    progressbar2==3.53.1 \
    h5netcdf==0.11.0 \
    geopy==2.2.0 \
    zarr==2.8.3 \
    tqdm==4.61.2 \
    rasterstats==0.15.0 \
    netCDF4==1.5.7 \
    utm==0.7.0 \
    wget==3.2 \
    selenium==3.141.0 \
    beautifulsoup4==4.9.3 \
    webdriver-manager==3.4.2 \
    torch==1.9.0 \
    torchvision==0.10.0 \
    oauth2client==4.1.3 \ 
    intake[complete]==0.6.2 \
    planetary-computer==0.4.2 \
    pystac==1.1.0 \
    pystac-client==0.3.0b1 \
    scikit-learn==0.24.2 \
    imageio==2.9.0 \
    lxml==4.6.3 \
    stackstac==0.2.1 \
    sphinx==4.2.0 \
    myst-parser==0.17.0 \
    sphinx-argparse==0.3.1 \
    sphinx-rtd-theme==1.0.0

# Install azcopy
RUN curl -L https://azcopyvnext.azureedge.net/release20210901/azcopy_linux_amd64_10.12.1.tar.gz > /tmp/azcopy.tar.gz
RUN tar -xzvf /tmp/azcopy.tar.gz && rm /tmp/azcopy.tar.gz
RUN ln -s /root/azcopy_linux_amd64_10.12.1/azcopy /usr/bin/azcopy

#add vim bindings
RUN mkdir -p $(jupyter --data-dir)/nbextensions &&\
     cd $(jupyter --data-dir)/nbextensions &&\
     git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding &&\
     jupyter nbextension enable vim_binding/vim_binding

#add blobfuse dependencies
#RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
#RUN dpkg -i packages-microsoft-prod.deb
#RUN apt-get update && apt-get install -y blobfuse
#RUN wget http://ftp.us.debian.org/debian/pool/main/i/icu/libicu63_63.2-3_amd64.deb &&\
#    dpkg -i libicu63_63.2-3_amd64.deb
#RUN apt install -y dotnet-runtime-2.1
