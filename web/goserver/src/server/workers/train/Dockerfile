############################
# STEP 1 build executable binary
############################
FROM golang:alpine AS builder

# Install git + SSL ca certificates.
# Git is required for fetching the dependencies.
# Ca-certificates is required to call HTTPS endpoints.
RUN apk update && apk add --no-cache git ca-certificates && update-ca-certificates

WORKDIR /go/src/server

# Fetch dependencies.
COPY web/goserver/src/server/go.mod .
COPY web/goserver/src/server/go.sum .
RUN go mod download
COPY web/goserver/src/server .
# Build the binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /go/bin/workers/train /go/src/server/workers/train/cmd

############################
# STEP 2 build PyTorch
############################
FROM nvidia/cuda:10.2-devel-ubuntu18.04
RUN apt update && apt install -y \
  apt-utils \
  build-essential \
  curl \
  git \
  libgdbm-dev \
  libglib2.0-0 \
  libffi-dev \
  libncurses5-dev \
  libnss3-dev \
  libreadline-dev \
  libxext6 \
  libxrender-dev \
  libsm6 \
  libssl-dev \
  python3 \
  python3-pip \
  unzip \
  wget \
  zlib1g-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sfn /usr/bin/python3 /usr/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

ENV FORCE_CUDA="1"

# Install packages for mmdetection
COPY ./pytorch_toolkit/object_detection/requirements.txt /tmp/ote/pytorch_toolkit/object_detection/requirements.txt
COPY ./external/mmdetection/requirements.txt /tmp/ote/external/mmdetection/requirements.txt
COPY ./external/mmdetection/requirements /tmp/ote/external/mmdetection/requirements
WORKDIR /tmp/ote/pytorch_toolkit/object_detection
RUN cat requirements.txt | xargs -n 1 -L 1 pip3 install
RUN pip3 install git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools

# gcc for cgo
RUN apt-get update && apt-get install -y --no-install-recommends \
		g++ \
		gcc \
		libc6-dev \
		make \
		pkg-config \
	&& rm -rf /var/lib/apt/lists/*


#FROM scratch
# Copy our static executable.
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /go/bin/workers/train /go/bin/workers/train

ENTRYPOINT rm -rf /ote/external/mmdetection/build && \
           cd /ote/external/mmdetection && \
           python setup.py develop && \
           /go/bin/workers/train
