
ARG UBUNTU_VER=20.04
FROM ubuntu:$UBUNTU_VER

ARG PYTHON_VER=3.9
ARG SOURCE=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install curl python$PYTHON_VER python$PYTHON_VER-dev python$PYTHON_VER-distutils g++ ffmpeg libsm6 libxext6 libgl1-mesa-glx -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python$PYTHON_VER get-pip.py && \
    rm get-pip.py

WORKDIR /training_extensions
COPY . /training_extensions

RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url $SOURCE && \
    pip install -e .[full]

CMD ["/bin/bash"]
