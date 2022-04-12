FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

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
        python3 \
        python3-dev \
        python3-venv \
        python3-opencv \
        python3-pip && \
    cd /usr/bin && \
    ln -s python3.8 python && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# OTE installation

RUN git clone https://github.com/openvinotoolkit/training_extensions.git && \
    cd  training_extensions && \
    git submodule update --init --recursive

ENV CUR_BE_PATH /training_extensions/external/mmdetection

RUN bash $CUR_BE_PATH/init_venv.sh /current_back_end
RUN /current_back_end/bin/pip install -e /training_extensions/ote_cli/ -c $CUR_BE_PATH/constraints.txt

ENTRYPOINT bash