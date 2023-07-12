#!/bin/bash

GPU_ID="all"
VER_CUDA="11.7.1"
TAG_RUNNER="latest"
ADDITIONAL_LABELS=""
MOUNT_PATH=""
DOCKER_REG_ADDR="local"
FIX_CPUS="0"
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
    -l|--labels)
      ADDITIONAL_LABELS="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--mount)
      MOUNT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--reg)
      DOCKER_REG_ADDR="$2"
      shift # past argument
      shift # past value
      ;;
    -f|--fix-cpus)
      FIX_CPUS="$2"
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
    USAGE: $0 <container-prefix> <github-token> <runner-prefix> [Options]
    Positional args
        <container-prefix>  Prefix to the ci container
        <github-token>      Github token string
        <runner-prefix>     Prefix to the actions-runner
    Options
        -g|--gpu-ids        GPU ID or IDs (comma separated) for runner or 'all'
        -c|--cuda           Specify CUDA version
        -t|--tag            Specify TAG for the CI container
        -l|--labels         Additional label string to set the actions-runner
        -m|--mount          Dataset root path to be mounted to the started container (absolute path)
        -r|--reg            Specify docker registry URL <default: local>
        -d|--debug          Flag to start debugging CI container
        -f|--fix-cpus       Specify the number of CPUs to set for the CI container
        -h|--help           Print this message
EndofMessage
exit 0
fi

CONTAINER_NAME=$1
GITHUB_TOKEN=$2
INSTANCE_NAME=$3
LABELS="self-hosted,Linux,X64"
ENV_FLAGS=""
MOUNT_FLAGS=""

if [ "$ADDITIONAL_LABELS" != "" ]; then
    LABELS="$LABELS,$ADDITIONAL_LABELS"
fi

echo "mount path option = $MOUNT_PATH"

if [ "$MOUNT_PATH" != "" ]; then
    ENV_FLAGS="-e CI_DATA_ROOT=/home/validation/data"
    MOUNT_FLAGS="-v $MOUNT_PATH:/home/validation/data:ro"
    LABELS="$LABELS,dmount"
fi

echo "env flags = $ENV_FLAGS, mount flags = $MOUNT_FLAGS"

if [ "$DEBUG_CONTAINER" = true ]; then
    CONTAINER_NAME="otx-ci-container-debug"
fi

CONTAINER_NAME="$CONTAINER_NAME"-${GPU_ID//,/_}
INSTANCE_NAME="$INSTANCE_NAME"-${GPU_ID//,/_}

echo "container name = $CONTAINER_NAME, instance name = $INSTANCE_NAME, labels = $LABELS"

docker inspect "$CONTAINER_NAME"; RET=$?

if [ $RET -eq 0 ]; then
    # if the named container exsiting, stop and remove it first
    docker stop "$CONTAINER_NAME"
    yes | docker rm "$CONTAINER_NAME"
fi

CPU_OPTIONS="--cpu-shares=1024"

if [ "$FIX_CPUS" != "0" ]; then
  CPU_OPTIONS="--cpus=$FIX_CPUS"
fi

if [ "$DEBUG_CONTAINER" = true ]; then
    # shellcheck disable=SC2086
    docker run -itd \
        --runtime=nvidia \
        --ipc=private \
        --shm-size=24g \
        "$CPU_OPTIONS" \
        --name "$CONTAINER_NAME" \
        -e NVIDIA_VISIBLE_DEVICES="$GPU_ID" \
        ${ENV_FLAGS} \
        ${MOUNT_FLAGS} \
        "$DOCKER_REG_ADDR"/ote/ci/cu"$VER_CUDA"/runner:"$TAG_RUNNER"; RET=$?

    if [ $RET -ne 0 ]; then
        echo "failed to start ci container. $RET"
        exit 1
    fi

    echo "Successfully started ci container for the debugging - $CONTAINER_NAME"
    exit 0
else
    # shellcheck disable=SC2086
    docker run -itd \
        --runtime=nvidia \
        --ipc=private \
        --shm-size=24g \
        "$CPU_OPTIONS" \
        --name "$CONTAINER_NAME" \
        -e NVIDIA_VISIBLE_DEVICES="$GPU_ID" \
        ${ENV_FLAGS} \
        ${MOUNT_FLAGS} \
        "$DOCKER_REG_ADDR"/ote/ci/cu"$VER_CUDA"/runner:"$TAG_RUNNER"; RET=$?

    if [ $RET -ne 0 ]; then
        echo "failed to start ci container. $RET"
        exit 1
    fi
fi


echo "Successfully started ci container - $CONTAINER_NAME"

docker exec -it "$CONTAINER_NAME" bash -c \
    "./actions-runner/config.sh  \
    --unattended \
    --url https://github.com/openvinotoolkit/training_extensions \
    --token $GITHUB_TOKEN \
    --name $INSTANCE_NAME \
    --labels $LABELS \
    --replace" ; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to configure the runner. $RET"
    docker exec -it "$CONTAINER_NAME" bash -c \
      "./actions-runner/config.sh --help"
    docker stop "$CONTAINER_NAME"
    yes | docker rm "$CONTAINER_NAME"
    exit 1
fi

docker exec -d "$CONTAINER_NAME" bash -c \
    "./actions-runner/run.sh" ; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to start actions runner. $RET"
    docker stop "$CONTAINER_NAME"
    yes | docker rm "$CONTAINER_NAME"
    exit 1
fi

echo "Successfully started actions runner"
