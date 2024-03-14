#!/bin/bash
http_proxy=${http_proxy:-}
https_proxy=${https_proxy:-}
no_proxy=${no_proxy:-}

docker build -t mlflow-tracker:v2.8.1 \
    --build-arg http_proxy="$http_proxy" \
    --build-arg https_proxy="$https_proxy" \
    --build-arg no_proxy="$no_proxy" .
