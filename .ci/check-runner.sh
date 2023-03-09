#!/bin/bash

RUNNER="0"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -r|--runner)
      RUNNER="1"
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

if [ "$#" -lt 1 ] || [ "$DEFAULT" == "yes" ]; then
cat << EndofMessage
    USAGE: $0 <container-name> [Options]
    Options
        -r|--runner         Check runner's log instead of Job one
        -h|--help           Print this message
EndofMessage
exit 0
fi

CONTAINER_NAME=$1

docker inspect "$CONTAINER_NAME"; RET=$?

if [ $RET -eq 0 ]; then
    if [ $RUNNER -eq "0" ]; then
        docker exec -it "$CONTAINER_NAME" bash -c \
            'logfile=$(find . -type f -name "Worker_*" | tail -1); tail -f $logfile'
    else
        docker exec -it "$CONTAINER_NAME" bash -c \
            'logfile=$(find . -type f -name "Runner_*" | tail -1); tail -f $logfile'
    fi
else
    echo "cannot find running container $CONTAINER_NAME"
    exit 1
fi
