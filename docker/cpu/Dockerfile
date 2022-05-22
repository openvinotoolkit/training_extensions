# docker build -t ote/mmdetection . --build-arg subproject=mmdetection

# Select one of OTE sub-prjects:
# - anomaly
# - deep-object-reid
# - mmdetection
# - mmsegmentation
# - model-preparation-algorithm

# Update packages
FROM intel/intel-optimized-pytorch:1.11.0-conda
RUN apt update && apt upgrade -y
RUN apt install -y python3-pip python3-venv
RUN apt install -y python3-opencv
RUN conda clean --all

# Create new user
RUN useradd -rm -d /home/tami -s /bin/bash -g root -G sudo -u 1001 tami
USER tami
RUN mkdir -p /home/tami/workspace
WORKDIR /home/tami/workspace

# Setup OpenVINO Training Extensions
RUN git clone https://github.com/openvinotoolkit/training_extensions.git
WORKDIR ./training_extensions
RUN git checkout -b v0.2.0
RUN git submodule update --init --recursive

ARG subproject
RUN ./external/"$subproject"/init_venv.sh "$subproject"_venv python3.8
RUN /bin/bash -c 'source "$subproject"_venv/bin/activate && pip3 install -e ote_cli/ && pip3 install notebook ipywidgets && deactivate'
WORKDIR /home/tami/workspace
