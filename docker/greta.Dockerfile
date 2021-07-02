FROM rocker/geospatial:4.0.3

### Greta and tensorflow stuff
ENV WORKON_HOME /opt/virtualenvs
ENV PYTHON_VENV_PATH $WORKON_HOME/r-tensorflow

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.7-dev python3.7-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.7 -m venv ${PYTHON_VENV_PATH}

RUN chown -R rstudio:rstudio ${WORKON_HOME}
ENV PATH ${PYTHON_VENV_PATH}/bin:${PATH}
## And set ENV for R! It doesn't read from the environment...
RUN echo "PATH=${PATH}" >> /usr/local/lib/R/etc/Renviron && \
    echo "WORKON_HOME=${WORKON_HOME}" >> /usr/local/lib/R/etc/Renviron && \
    echo "RETICULATE_PYTHON_ENV=${PYTHON_VENV_PATH}" >> /usr/local/lib/R/etc/Renviron

## Because reticulate hardwires these PATHs...
RUN ln -s ${PYTHON_VENV_PATH}/bin/pip /usr/local/bin/pip && \
    ln -s ${PYTHON_VENV_PATH}/bin/virtualenv /usr/local/bin/virtualenv

## install as user to avoid venv issues later
USER rstudio

RUN pip3 install wheel

RUN pip3 install \
    h5py==3.1.0 \
    numpy==1.19.2 \
    six==1.15.0 \
    pyyaml==3.13 \
    requests==2.21.0 \
    Pillow==5.4.1 \
    tensorflow==1.14.0 \
    tensorflow-probability==0.7.0 \
    keras==2.2.4 \
    --no-cache-dir

USER root
RUN install2.r reticulate tensorflow keras greta

RUN R -e 'devtools::install_github("r-barnes/dggridR")'

# Install package deps
RUN install2.r \
    bayesplot \
	doParallel \
	foreach \
	ggthemes \
	RhpcBLASctl
