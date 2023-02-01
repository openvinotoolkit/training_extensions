#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
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

if [ "$#" -lt 2 ] || [ "$DEFAULT" == "yes" ]; then
cat << EndofMessage
    USAGE: $0 <container-name> <github-token> [Options]
    Options
        -h|--help           Print this message
EndofMessage
exit 0
fi

CONTAINER_NAME=$1
GITHUB_TOKEN=$2

docker inspect "$CONTAINER_NAME"; RET=$?

if [ $RET -eq 0 ]; then
    docker exec -it "$CONTAINER_NAME" bash -c \
        "./actions-runner/config.sh remove \
        --token $GITHUB_TOKEN" ; RET=$?

    if [ $RET -ne 0 ]; then
        echo "failed to stop the runner. $RET"
        exit 1
    fi

    # stop and remove ci container
    docker stop "$CONTAINER_NAME"
    yes | docker rm "$CONTAINER_NAME"
else
    echo "cannot find running container $CONTAINER_NAME"
    exit 1
fi