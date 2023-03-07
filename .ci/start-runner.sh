#!/bin/bash

GPU_ID="all"
VER_CUDA="11.1.1"
TAG_RUNNER="latest"
DEBUG_CONTAINER=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -g|--gpu-ids)
      GPU_ID="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--cuda)
      VER_CUDA="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--tag)
      TAG_RUNNER="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--debug)
      DEBUG_CONTAINER=true
      shift # past argument
      ;;
    -h|--help)
      DEFAULT="yes"
      break
      shift # past argument
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

if [[ "$#" -lt 3 ||  "$DEFAULT" == "yes" ]] && [ $DEBUG_CONTAINER = false ]; then
cat << EndofMessage
    USAGE: $0 <container-name> <github-token> <codacy-token> [Options]
    Options
        -g|--gpu-ids        GPU ID or IDs (comma separated) for runner or 'all'
        -c|--cuda           Specify CUDA version
        -t|--tag            Specify TAG for the CI container
        -d|--debug          Flag to start debugging CI container
        -h|--help           Print this message
EndofMessage
exit 0
fi

CONTAINER_NAME=$1
GITHUB_TOKEN=$2
CODACY_TOKEN=$3

if [ "$DEBUG_CONTAINER" = true ]; then
    CONTAINER_NAME="otx-ci-container-debug"
fi

docker inspect "$CONTAINER_NAME"; RET=$?

if [ $RET -eq 0 ]; then
    # if the named container exsiting, stop and remove it first
    docker stop "$CONTAINER_NAME"
    yes | docker rm "$CONTAINER_NAME"
fi


if [ "$DEBUG_CONTAINER" = true ]; then
    docker run -itd \
        --env NVIDIA_VISIBLE_DEVICES="$GPU_ID" \
        --runtime=nvidia \
        --ipc=host \
        --cpus=40 \
        --name "$CONTAINER_NAME" \
        registry.toolbox.iotg.sclab.intel.com/ote/ci/cu"$VER_CUDA"/runner:"$TAG_RUNNER"; RET=$?

    if [ $RET -ne 0 ]; then
        echo "failed to start ci container. $RET"
        exit 1
    fi

    echo "Successfully started ci container for the debugging - $CONTAINER_NAME"
    exit 0
else
    docker run -itd \
        --env NVIDIA_VISIBLE_DEVICES="$GPU_ID" \
        --runtime=nvidia \
        --ipc=host \
        --cpus=40 \
        --name "$CONTAINER_NAME" \
        registry.toolbox.iotg.sclab.intel.com/ote/ci/cu"$VER_CUDA"/runner:"$TAG_RUNNER"; RET=$?

    if [ $RET -ne 0 ]; then
        echo "failed to start ci container. $RET"
        exit 1
    fi
fi


echo "Successfully started ci container - $CONTAINER_NAME"

docker exec -it "$CONTAINER_NAME" bash -c \
    "./actions-runner/config.sh  \
    --url https://github.com/openvinotoolkit/training_extensions \
    --token $GITHUB_TOKEN" ; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to configure the runner. $RET"
    exit 1
fi

docker exec -d "$CONTAINER_NAME" bash -c \
    "export CODACY_PROJECT_TOKEN=$CODACY_TOKEN && \
    ./actions-runner/run.sh" ; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to start actions runner. $RET"
    exit 1
fi

echo "Successfully started actions runner"
