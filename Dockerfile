#########################################################
## Python Environment with CUDA
#########################################################
ARG http_proxy
ARG https_proxy
ARG no_proxy

FROM nvidia/cuda:10.2-devel-ubuntu18.04 AS python_base_cuda
LABEL MAINTAINER="OpenVINO Training Extensions Development Team"

# Setup proxies

ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy

# Update system and install wget
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        wget \
        ffmpeg \
        libpython3.8 \
        git \
        sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install python=3.8

#########################################################
## OTX Development Env
#########################################################

FROM python_base_cuda as otx_development_env

# Install all OTX requirements. Serves as a way to cache the requirements

COPY ./requirements/base.txt /tmp/otx/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/base.txt

COPY ./requirements/anomaly.txt /tmp/otx/requirements/anomaly.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/anomaly.txt

COPY ./requirements/dev.txt /tmp/otx/requirements/dev.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/dev.txt

COPY ./requirements/docs.txt /tmp/otx/requirements/docs.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/docs.txt

COPY ./requirements/openvino.txt /tmp/otx/requirements/openvino.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/openvino.txt

# Install other requirements related to development
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        nodejs \
        npm \
        ruby && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install OTX
COPY . /otx
WORKDIR /otx
RUN pip install -e .