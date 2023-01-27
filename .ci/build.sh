#!/bin/bash

VER_CUDA="11.1.1"
ACTIONS_RUNNER_URL="https://github.com/actions/runner/releases/download/v2.299.1/actions-runner-linux-x64-2.299.1.tar.gz"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -p|--push)
      PUSH="yes"
      shift # past argument
      ;;
    -u|--url)
      ACTIONS_RUNNER_URL="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--cuda)
      VER_CUDA="$2"
      shift # past argument
      shift # past value
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

if [ "$#" -lt 1 ] || [ "$DEFAULT" == "yes" ]; then
cat << EndofMessage
    USAGE: $0 <tag> [Options]
    Options
        -p|--push           Push built image(s) to registry
        -u|--url            url to get Github actions-runner package
        -c|--cuda           Specify CUDA version
        -h|--help           Print this message
EndofMessage
exit 0
fi

TAG=$1

docker build -f .ci/Dockerfile \
--build-arg HTTP_PROXY="${http_proxy:?}" \
--build-arg HTTPS_PROXY="${https_proxy:?}" \
--build-arg NO_PROXY="${no_proxy:?}" \
--build-arg ver_cuda="$VER_CUDA" \
--build-arg action_runner_url="$ACTIONS_RUNNER_URL" \
--tag registry.toolbox.iotg.sclab.intel.com/ote/ci/cu"$VER_CUDA"/runner:"$TAG" \
--tag registry.toolbox.iotg.sclab.intel.com/ote/ci/cu"$VER_CUDA"/runner:latest .; RET=$?


if [ $RET -ne 0 ]; then
    echo "failed to build a 'ote/ci/cu$VER_CUDA/runner' image. $RET"
    exit 1
fi

echo "Successfully built docker image."

if [ "$PUSH" == "yes" ]; then
    docker push registry.toolbox.iotg.sclab.intel.com/ote/ci/cu"$VER_CUDA"/runner:"$TAG"; RET=$?
    if [ $RET -ne 0 ]; then
        echo "failed to push a docker image to registry. $RET"
        exit 1
    fi
    docker push registry.toolbox.iotg.sclab.intel.com/ote/ci/cu"$VER_CUDA"/runner:latest; RET=$?
    if [ $RET -ne 0 ]; then
        echo "failed to push a docker image to registry. $RET"
        exit 1
    fi
else
    echo "Newly built image was not pushed to the registry. use '-p|--push' option to push image."
fi


