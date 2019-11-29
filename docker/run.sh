#!/bin/bash

set -e

display_usage() {
  echo
  echo "Usage: $0 [cpu,gpu]"
  echo
}


TARGET="${1}"
IMAGE_NAME="ote_base"
IMAGE_FULL_NAME="${IMAGE_NAME}:${TARGET}"
CONTAINER_NAME="ote_${TARGET}_$(date +%s)"

case "$TARGET" in
    gpu)

        EXEC_BIN="nvidia-docker"
        ;;
    cpu)
        EXEC_BIN="docker"
        ;;
    *)
        display_usage
        exit 1
        ;;
  esac

$EXEC_BIN run \
    --user="$(id -u):$(id -g)" \
    --env http_proxy="${http_proxy}" \
    --env https_proxy="${https_proxy}" \
    --env no_proxy="${no_proxy}" \
    --name "${CONTAINER_NAME}" \
    --volume "$(git rev-parse --show-toplevel):/workspace" \
    --interactive \
    --tty \
    --rm \
    "${IMAGE_FULL_NAME}" \
    bash
