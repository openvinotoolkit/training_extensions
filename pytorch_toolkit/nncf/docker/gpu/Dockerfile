FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        git && \
rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        zip \
        unzip \
        nano \
        openssh-server \
        openssh-client \
        sudo \
    && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends  python3-dev  && \
    cd /usr/bin && \
    ln -s python3.5 python && \
    curl https://bootstrap.pypa.io/get-pip.py | python \
    && \
    rm -rf /var/lib/apt/lists/* 

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ARG https_proxy
ARG no_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy

ENTRYPOINT cd /home/nncf/nn-compression-framework/ && python setup.py install  && bash
