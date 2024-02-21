#!/bin/bash

PYTHON_VER="3.9"

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -p|--python)
      PYTHON_VER="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      DEFAULT="yes"
      break
      shift # past argument
      ;;
  esac
done

if [ "$DEFAULT" == "yes" ]; then
cat << EndofMessage
    USAGE: $0 [Options]
    Options
        -p|--python         Specify Python version
        -h|--help           Print this message
EndofMessage
exit 0
fi

docker build -f docker/Dockerfile.cpu \
--build-arg HTTP_PROXY="${http_proxy:?}" \
--build-arg HTTPS_PROXY="${https_proxy:?}" \
--build-arg NO_PROXY="${no_proxy:?}" \
--build-arg python_ver="$PYTHON_VER" \
--tag otx/cpu/python"$PYTHON_VER":latest .; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to build a 'otx/cpu/python$PYTHON_VER:latest' image. $RET"
    exit 1
fi

echo "Successfully built docker image."
