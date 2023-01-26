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

# added ninja-build for using CppExtension in the torch
# hadolint ignore=DL3008
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

RUN mkdir actions-runner
WORKDIR /home/validation/actions-runner
ARG action_runner_url
# download actions-runner and extract it
RUN curl -o actions-runner-linux-x64.tar.gz -L ${action_runner_url} && \
    tar xzf ./actions-runner-linux-x64.tar.gz && \
    rm actions-runner-linux-x64.tar.gz

WORKDIR /home/validation
