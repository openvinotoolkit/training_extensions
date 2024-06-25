ARG http_proxy
ARG https_proxy
ARG no_proxy

FROM python:3.10-slim-bullseye@sha256:f9d945b64f64eea6c7976c8ab75dc49df24801f822ce1cd24680d2fc9598feeb

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY requirements.in /tmp/requirements.in

RUN pip-compile --generate-hashes --output-file=/tmp/requirements.txt /tmp/requirements.in

RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt /tmp/requirements.in
