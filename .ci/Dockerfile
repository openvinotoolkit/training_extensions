#########################################################
## Python Environment with CUDA
#########################################################
ARG ver_cuda="11.7.1"

FROM nvidia/cuda:${ver_cuda}-devel-ubuntu20.04@sha256:f663a1cf01a46daa469c75bf246ac00098bd5179aff2c75367c44f475cd4c8f4 AS python_base_cuda
LABEL maintainer="OpenVINO Training Extensions Development Team"

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG action_runner_url
ARG uid
ARG gid

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
        libpython3.10 \
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
    # Create a non-root user with having given UID & GID
    groupadd -r -g ${gid} validation && \
    useradd -l -r -m validation -g ${gid} -u ${uid} && \
    echo "${gid}:${uid}"

USER validation

WORKDIR /home/validation


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
