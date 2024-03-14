#!/bin/bash
# shellcheck disable=SC2154

OTX_VERSION=$(python -c 'import otx; print(otx.__version__)')
THIS_DIR=$(dirname "$0")

echo "Build OTX ${OTX_VERSION} CUDA Docker image..."
docker build \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    --target cuda \
    -t "otx:${OTX_VERSION}-cuda" \
    -f "${THIS_DIR}/Dockerfile.cuda" "${THIS_DIR}"/..

echo "Build OTX ${OTX_VERSION} CUDA pretrained-ready Docker image..."
docker build \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    --target cuda_pretrained_ready \
    -t "otx:${OTX_VERSION}-cuda-pretrained-ready" \
    -f "${THIS_DIR}/Dockerfile.cuda" "${THIS_DIR}"/..
