#!/bin/bash

set -e

display_usage() {
  echo
  echo "Usage: $0 [cpu,gpu]"
  echo
}


TARGET="${1:-gpu}"
IMAGE_NAME="ote_base"
IMAGE_FULL_NAME="${IMAGE_NAME}:${TARGET}"


case "$TARGET" in
    gpu)
        CUDA_VERSION="${CUDA_VERSION:-10.0}"
        BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu18.04"
        EXEC_BIN="nvidia-docker"
        CPU_ONLY="false"
        ;;
    cpu)
        BASE_IMAGE="ubuntu:18.04"
        EXEC_BIN="docker"
        CPU_ONLY="true"
        ;;
    *)
        display_usage
        exit 1
        ;;
  esac

OPENVINO_LINK="http://registrationcenter-download.intel.com/akdlm/irc_nas/15944/l_openvino_toolkit_p_2019.3.334.tgz"

echo ""
echo "Base name:     ${BASE_IMAGE}"
echo "Image name:    ${IMAGE_FULL_NAME}"
echo "OpenVINO link: ${DOWNLOAD_LINK}"
echo ""

# shellcheck disable=SC2154
$EXEC_BIN build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg CPU_ONLY="${CPU_ONLY}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    --build-arg OPENVINO_LINK="${OPENVINO_LINK}" \
    -t "${IMAGE_FULL_NAME}" \
    -f Dockerfile \
    .
