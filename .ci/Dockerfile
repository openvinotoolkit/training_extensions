#########################################################
## Python Environment with CUDA
#########################################################
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

FROM nvidia/cuda:10.2-devel-ubuntu18.04 AS python_base_cuda
LABEL MAINTAINER="OpenVINO Training Extensions Development Team"

# Setup proxies

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY

# Update system and install wget
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        wget \
        ffmpeg \
        libpython3.8 \
        git \
        curl \
        nodejs \
        npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m validation
USER validation

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet && \
    bash /home/validation/miniconda.sh -b -p /home/validation/conda && \
    rm ~/miniconda.sh
ENV PATH "/home/validation/conda/bin:${PATH}"
RUN conda install python=3.8

#########################################################
## OTX Development Env
#########################################################

FROM python_base_cuda as otx_development_env

# Install all OTX requirements. Serves as a way to cache the requirements

COPY ./requirements/anomaly.txt /tmp/otx/requirements/anomaly.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/anomaly.txt

COPY ./requirements/base.txt /tmp/otx/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/base.txt

COPY ./requirements/dev.txt /tmp/otx/requirements/dev.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/dev.txt

COPY ./requirements/docs.txt /tmp/otx/requirements/docs.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/docs.txt

COPY ./requirements/openvino.txt /tmp/otx/requirements/openvino.txt
RUN pip install --no-cache-dir -r /tmp/otx/requirements/openvino.txt

WORKDIR /home/validation