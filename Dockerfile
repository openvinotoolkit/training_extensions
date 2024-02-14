
ARG UBUNTU_VER=20.04@sha256:a4fab1802f08df089c4b2e0a1c8f1a06f573bd1775687d07fef4076d3a2e4900
FROM ubuntu:$UBUNTU_VER

ARG PYTHON_VER=3.9
ARG SOURCE=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl python$PYTHON_VER python$PYTHON_VER-dev python$PYTHON_VER-distutils g++ ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python$PYTHON_VER get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /training_extensions
COPY . /training_extensions

# hadolint ignore=SC2102
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url $SOURCE && \
    pip install --no-cache-dir -e .[full]

CMD ["/bin/bash"]
