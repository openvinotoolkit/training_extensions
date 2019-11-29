ARG BASE_IMAGE=nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
FROM ${BASE_IMAGE}

ENV TERM=xterm
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        build-essential \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        virtualenv \
        libgomp1 \
        libglib2.0-0 \
        libgtk-3-dev \
        lsb-release \
        libsm6 \
        libxtst6 \
        libturbojpeg \
        2to3 \
        protobuf-compiler \
        cpio \
        ffmpeg \
        unzip \
        wget \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

ARG OPENVINO_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/15944/l_openvino_toolkit_p_2019.3.334.tgz

RUN mkdir -p /tmp/openvino_installer && cd /tmp/openvino_installer && \
    wget -e no_proxy="" -c $OPENVINO_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    rm l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    bash ./install.sh -s silent.cfg --accept_eula --components "intel-openvino-ie-sdk-ubuntu-bionic__x86_64;intel-openvino-ie-rt-cpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-gpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-vpu-ubuntu-bionic__x86_64;intel-openvino-model-optimizer__x86_64;intel-openvino-opencv-lib-ubuntu-bionic__x86_64;intel-openvino-mediasdk__x86_64" && \
    cd /opt/intel/openvino/install_dependencies/ && \
    dpkg -i intel-*.deb && \
    ldconfig && \
    rm -rf /tmp/openvino_installer

RUN echo 'export PS1="\w\$ "' >> /etc/bash.bashrc

ARG CPU_ONLY=false
ENV CPU_ONLY=${CPU_ONLY}

WORKDIR /workspace
