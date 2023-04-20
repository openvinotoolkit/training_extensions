
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install curl python3.9 python3.9-dev python3.9-distutils git g++ ffmpeg libsm6 libxext6 libgl1-mesa-glx -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

COPY . training_extensions/

RUN cd training_extensions && \
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .[full]

CMD ["/bin/bash"]
