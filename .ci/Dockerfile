#########################################################
## Python Environment with CUDA
#########################################################
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG action_runner_url
ARG ver_cuda="11.1.1"

FROM nvidia/cuda:${ver_cuda}-devel-ubuntu20.04 AS python_base_cuda
LABEL maintainer="OpenVINO Training Extensions Development Team"

# Setup proxies

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV DEBIAN_FRONTEND="noninteractive"

# Update system and install dependencies
# RUN apt-get update && apt-get install --no-install-recommends -y \
#         software-properties-common=0.99.9.8 \
#         wget=1.20.3-1ubuntu2 \
#         ffmpeg=7:4.2.7-0ubuntu0.1 \
#         libpython3.8=3.8.10-0ubuntu1\~20.04.6 \
#         curl=7.68.0-1ubuntu2.15 \
#         nodejs=10.19.0\~dfsg-3ubuntu1 \
#         unzip=6.0-25ubuntu1.1 \
#         npm=6.14.4+ds-1ubuntu2 \
#         ruby-dev=1:2.7+1 \
#         git=1:2.25.1-1ubuntu3.7 \
#         cron=3.0pl1-136ubuntu1 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# added ninja-build for using CppExtension in the torch
RUN apt-get update && apt-get install --no-install-recommends -y \
        software-properties-common \
        wget \
        ffmpeg \
        libpython3.8 \
        curl \
        nodejs \
        unzip \
        npm \
        ruby-dev \
        git \
        cron \
        ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set /tmp folder cleaning schedule at 7PM every day which is older than a day
RUN echo "0 19 * * * find /tmp/* -mtime +1 -exec rm -rf {} \;" >> ./cron_clean_tmp.txt && \
    crontab cron_clean_tmp.txt && \
    # Create a non-root user
    useradd -m validation

USER validation

WORKDIR /home/validation

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

# # Install all OTX requirements. Serves as a way to cache the requirements

# COPY ./requirements/anomaly.txt /tmp/otx/requirements/anomaly.txt
# COPY ./requirements/base.txt /tmp/otx/requirements/base.txt
# COPY ./requirements/dev.txt /tmp/otx/requirements/dev.txt
# COPY ./requirements/docs.txt /tmp/otx/requirements/docs.txt
# COPY ./requirements/openvino.txt /tmp/otx/requirements/openvino.txt

# RUN pip install --no-cache-dir -r /tmp/otx/requirements/anomaly.txt \
#     -r /tmp/otx/requirements/base.txt \
#     -r /tmp/otx/requirements/dev.txt \
#     -r /tmp/otx/requirements/docs.txt \
#     -r /tmp/otx/requirements/openvino.txt && \
#     mkdir actions-runner

# RUN pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111 \
#     && mkdir actions-runner
RUN mkdir actions-runner

WORKDIR /home/validation/actions-runner

ARG action_runner_url
RUN curl -o actions-runner-linux-x64.tar.gz -L ${action_runner_url} && \
    tar xzf ./actions-runner-linux-x64.tar.gz && \
    rm actions-runner-linux-x64.tar.gz

WORKDIR /home/validation
