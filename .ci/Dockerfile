#########################################################
## Python Environment with CUDA
#########################################################
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

FROM nvidia/cuda:10.2-devel-ubuntu18.04 AS python_base_cuda
LABEL maintainer="OpenVINO Training Extensions Development Team"

# Setup proxies

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV DEBIAN_FRONTEND="noninteractive"

# Update system and install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    software-properties-common=0.96.24.32.18 \
        wget=1.19.4-1ubuntu2.2 \
        ffmpeg=7:3.4.11-0ubuntu0.1 \
        libpython3.8=3.8.0-3ubuntu1\~18.04.2 \
        curl=7.58.0-2ubuntu3.20 \
        nodejs=8.10.0\~dfsg-2ubuntu0.4 \
        unzip=6.0-21ubuntu1.1 \
        npm=3.5.2-0ubuntu4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install latest git for github actions
RUN apt-get update && \
    add-apt-repository ppa:git-core/ppa &&\
    apt-get install --no-install-recommends -y git=1:2.37.3-0ppa1\~ubuntu18.04.1 &&\
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
