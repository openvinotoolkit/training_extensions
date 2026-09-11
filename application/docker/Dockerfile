# syntax=docker/dockerfile:1.7
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# If set to 1, uninstalls ultralytics and its AGPL-licensed dependencies after installation.
ARG EXCLUDE_AGPL_MODELS=0

##############
# Backend base
##############
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS backend-base

# uv / uvx
COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/

# Increase HTTP timeout for large packages (e.g. torch)
ENV UV_HTTP_TIMEOUT=300
# Compile bytecode at install time for faster container startup
ENV UV_COMPILE_BYTECODE=1
# Use copy instead of hardlinks (cache mount is on a different filesystem)
ENV UV_LINK_MODE=copy
# Don't download a Python interpreter; use the one from the base image
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /application/backend

# Build deps (for compiling / Rust-based crates)
# TODO: remove 'git' once all dependencies are on PyPI
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update  \
    && apt-get install -y --no-install-recommends \
        build-essential=12.12 \
        curl=8.14.1-2* \
        git=1:2.47.3-0*

# Install Rust toolchain
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

##########
# CPU Base
##########
FROM backend-base AS cpu-base
ARG EXCLUDE_AGPL_MODELS

# Step 1: Install all third-party dependencies (cache key = uv.lock + pyproject.toml + version only).
# --no-install-project skips building/installing the local project and its path deps (getitune),
# so changes to library/ source code do NOT invalidate this (slow) layer.
# Note: __init__.py is needed because getitune uses dynamic versioning (attr = "getitune.__version__").
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library/pyproject.toml,target=/library/pyproject.toml \
    --mount=type=bind,source=library/src/getitune/__init__.py,target=/library/src/getitune/__init__.py \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-workspace \
    --no-install-package av --extra mqtt --extra cpu

# Step 2: Build and install the local getitune package + the project itself.
# This layer is invalidated by library source changes, but is fast since all
# third-party deps are already installed from step 1.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --no-install-package av --extra mqtt --extra cpu

# Optionally uninstall ultralytics and its AGPL-licensed dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$EXCLUDE_AGPL_MODELS" = "1" ]; then \
        uv pip uninstall ultralytics ultralytics-thop nvidia-ml-py; \
    fi

###########
# CUDA Base
###########
FROM backend-base AS cuda-base
ARG EXCLUDE_AGPL_MODELS

# Step 1: Install all third-party dependencies (cache key = uv.lock + pyproject.toml + version only).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library/pyproject.toml,target=/library/pyproject.toml \
    --mount=type=bind,source=library/src/getitune/__init__.py,target=/library/src/getitune/__init__.py \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-workspace \
    --no-install-package av --extra mqtt --extra cuda

# Step 2: Build and install the local getitune package + the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --no-install-package av --extra mqtt --extra cuda

# Optionally uninstall ultralytics and its AGPL-licensed dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$EXCLUDE_AGPL_MODELS" = "1" ]; then \
        uv pip uninstall ultralytics ultralytics-thop nvidia-ml-py; \
    fi

##########
# XPU Base
##########
FROM backend-base AS xpu-base
ARG EXCLUDE_AGPL_MODELS

# Step 1: Install all third-party dependencies (cache key = uv.lock + pyproject.toml + version only).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library/pyproject.toml,target=/library/pyproject.toml \
    --mount=type=bind,source=library/src/getitune/__init__.py,target=/library/src/getitune/__init__.py \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-workspace \
    --no-install-package av --extra mqtt --extra xpu

# Step 2: Build and install the local getitune package + the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --no-install-package av --extra mqtt --extra xpu

# Optionally uninstall ultralytics and its AGPL-licensed dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$EXCLUDE_AGPL_MODELS" = "1" ]; then \
        uv pip uninstall ultralytics ultralytics-thop nvidia-ml-py; \
    fi

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:191c9f0080fcbbc6547a85dc0ff7988072214a355aabdc1d2ec55a7dae5eea8a AS web-ui-base

WORKDIR /home/application/ui/

COPY --link application/ui/package.json ./
COPY --link application/ui/package-lock.json ./
RUN npm ci --audit=false

COPY --link application/ui/tsconfig.json ./
COPY --link application/ui/rsbuild.config.ts ./
COPY --link application/ui/src/ src/
COPY --link application/ui/public/ public/
COPY --link application/ui/vendor/ vendor/

########
# Web UI
########
FROM web-ui-base AS web-ui

ENV ASSET_PREFIX="/html"

RUN npm run build

#===========================================================================
# pyav + ffmpeg
# ===========================================================================
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS ffmpeg-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Pin refs/versions if you want reproducible builds.
ARG LIBVPL_REF=778a66d6c6537f08eabb91955dbbf1bce3812894
ARG FFMPEG_REF=9dd0fe0225c2c3406ca49430a74c8660b1713889
ARG PYAV_VERSION=16.1.0
ARG OPENH264_REF=e3f5b10438e2bacc155cf54578222bd4236c9f06
ARG SVTAV1_REF=349f4eb9c369f68d463029d1cd37551c90b7de51

# Install build and runtime dependencies.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash=5.2* \
    ca-certificates=202* \
    curl=8.14.1-2* \
    git=1:2.47.3-0* \
    build-essential=12.12 \
    cmake=3.31* \
    meson=1.7* \
    ninja-build=1.12* \
    pkg-config=1.8* \
    yasm=1.3* \
    nasm=2.16* \
    libdrm-dev=2.4* \
    libva-dev=2.22* \
    libnuma-dev=2.0* \
    libvpx-dev=1.15.0* \
    && rm -rf /var/lib/apt/lists/*

# Python packaging tools for building wheels.
RUN python -m pip install --no-cache-dir \
    "pip==25.*" \
    "setuptools==75.*" \
    "wheel==0.45.*" \
    "Cython==3.0.*"

# Build/install oneVPL dispatcher + headers.
ENV VPL_INSTALL_DIR=/opt/vpl
WORKDIR /tmp

RUN git init libvpl && \
    git -C libvpl remote add origin https://github.com/intel/libvpl.git && \
    git -C libvpl fetch --depth 1 origin "${LIBVPL_REF}" && \
    git -C libvpl checkout FETCH_HEAD && \
    cmake -B libvpl/build -S /tmp/libvpl \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${VPL_INSTALL_DIR}" && \
    cmake --build libvpl/build -j"$(nproc)" && \
    cmake --install libvpl/build

# Make custom libvpl and FFmpeg visible to later build steps.
ENV FFMPEG_INSTALL_DIR=/opt/ffmpeg-vpl
ENV PKG_CONFIG_PATH="/opt/vpl/lib/pkgconfig:/opt/vpl/lib64/pkgconfig:/opt/ffmpeg-vpl/lib/pkgconfig:/usr/local/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/opt/vpl/lib:/opt/vpl/lib64:/opt/ffmpeg-vpl/lib:/usr/local/lib"
ENV PATH="/opt/ffmpeg-vpl/bin:${PATH}"

# Optional software fallback encoders.
# Remove these two blocks if you do not want libopenh264 / libsvtav1.

RUN git init openh264 && \
    git -C openh264 remote add origin https://github.com/cisco/openh264.git && \
    git -C openh264 fetch --depth 1 origin "${OPENH264_REF}" && \
    git -C openh264 checkout FETCH_HEAD && \
    make -C openh264 -j"$(nproc)" && \
    make -C openh264 install && \
    ldconfig

RUN git init svtav1 && \
    git -C svtav1 remote add origin https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    git -C svtav1 fetch --depth 1 origin "${SVTAV1_REF}" && \
    git -C svtav1 checkout FETCH_HEAD && \
    cmake -B svtav1/build -S /tmp/svtav1 -DCMAKE_BUILD_TYPE=Release && \
    cmake --build svtav1/build -j"$(nproc)" && \
    cmake --install svtav1/build && \
    ldconfig

# Build FFmpeg with libvpl enabled
RUN git init ffmpeg && \
    git -C ffmpeg remote add origin https://github.com/FFmpeg/FFmpeg.git && \
    git -C ffmpeg fetch --depth 1 origin "${FFMPEG_REF}" && \
    git -C ffmpeg checkout FETCH_HEAD

WORKDIR /tmp/ffmpeg

# Configure and compile inside the repository context
RUN ./configure \
      --prefix="${FFMPEG_INSTALL_DIR}" \
      --extra-cflags="-I${VPL_INSTALL_DIR}/include" \
      --extra-ldflags="-L${VPL_INSTALL_DIR}/lib -L${VPL_INSTALL_DIR}/lib64" \
      --extra-libs="-lpthread -lm" \
      --enable-libvpl \
      --enable-vaapi \
      --enable-shared \
      --enable-libsvtav1 \
      --enable-libopenh264 \
      --enable-libvpx && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# Verify FFmpeg exposes QSV encoders.
RUN ffmpeg -version && \
    ffmpeg -hide_banner -encoders | grep qsv && \
    ffmpeg -hide_banner -h encoder=h264_qsv > /dev/null

# Build PyAV wheel from source against the FFmpeg we just built.
WORKDIR /build
RUN python -m pip wheel --no-binary av "av==${PYAV_VERSION}" -w /dist

#########################
# Base for CPU, GPU and XPU runtime images
#########################
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS runtime-base

LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/training_extensions"
LABEL org.opencontainers.image.description="Geti – train, optimize and deploy deep-learning models"
LABEL org.opencontainers.image.revision="${GIT_SHA}"

ARG UID=10001
ARG USER_NAME="non-root"

ENV STATIC_FILES_DIR="/application/html"
ENV DATA_DIR="/application/data"
ENV LOG_DIR="/application/logs"
ENV PYTHONPATH=/application/backend
ENV PYTHONUNBUFFERED=1

# Create secure non-root user, add it to the 'video' group (for /dev/videoN access),
# and remove unused package managers to reduce image size
RUN groupadd -f video && \
    useradd -l -u ${UID} -G video ${USER_NAME} && \
    pip uninstall -y setuptools pip wheel && \
    rm -rf /root/.cache/pip

# Create directory and set ownership so mounting a volume works correctly
RUN mkdir -p ${DATA_DIR} && chown ${UID}:${UID} ${DATA_DIR}
RUN mkdir -p ${LOG_DIR} && chown ${UID}:${UID} ${LOG_DIR}

# Common system deps (CPU/GPU/XPU all need these)
# TODO: remove 'git' once all dependencies are available on PyPI
# libv4l-dev: V4L2 userspace libraries required by OpenCV to capture from USB cameras (/dev/videoN)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0* \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0t64=2.84.4-3~deb13u3 \
        libglx-mesa0=25.0.* \
        libv4l-0t64=1.30.1-1 \
        libva2=2.22.* \
        libva-drm2=2.22.* \
        libvpx9=1.15.0* \
        openssl=3.5.7*

# Place the venv at the front of PATH so we can call python directly (no uv needed at runtime)
ENV PATH="/application/backend/.venv/bin:${PATH}"

# Create directories and set ownership for the non-root user.
RUN mkdir -p /application/backend/app ${STATIC_FILES_DIR} /home/${USER_NAME} && \
    chown -R ${UID}:${UID} \
    /application \
    /home/${USER_NAME}

COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /bin/uv

# Copy custom FFmpeg shared libs and PyAV wheel
COPY --link --from=ffmpeg-builder /dist /dist
COPY --link --from=ffmpeg-builder /opt/vpl /opt/vpl
COPY --link --from=ffmpeg-builder /opt/ffmpeg-vpl /opt/ffmpeg-vpl
COPY --link --from=ffmpeg-builder /usr/local/lib/libSvtAv1* /usr/local/lib/
COPY --link --from=ffmpeg-builder /usr/local/lib/libopenh264* /usr/local/lib/

ENV LD_LIBRARY_PATH="/opt/vpl/lib:/opt/vpl/lib64:/opt/ffmpeg-vpl/lib:/usr/local/lib"
ENV PATH="/opt/ffmpeg-vpl/bin:${PATH}"

WORKDIR /application/backend
COPY --link --chmod=755 application/docker/entrypoint.sh /entrypoint.sh

USER ${USER_NAME}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app/main.py"]

######################
# Cleanup non-redistributable CUDA components per NVIDIA EULA
######################

FROM cuda-base AS cuda-clean

RUN set -eu; \
    CLEANUP_ROOT="$("/application/backend/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    [ -n "$CLEANUP_ROOT" ] && [ -d "$CLEANUP_ROOT" ]; \
    echo "Removing non-redistributable CUDA components per NVIDIA EULA from ${CLEANUP_ROOT}..." && \
    # Remove headers and include directories (non-redistributable)
    find "$CLEANUP_ROOT" -type d \( \
    -name "include" -o \
    -name "headers" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove NVIDIA package includes specifically
    find "$CLEANUP_ROOT" -type d -path "*/nvidia/*/include" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove static libraries (typically non-redistributable)
    find "$CLEANUP_ROOT" -name "*.a" \( -path "*cuda*" -o -path "*nvidia*" \) -delete 2>/dev/null || true && \
    # Remove non-redistributable headers and binaries from triton nvidia backend
    find "$CLEANUP_ROOT" -type d -path "*/triton/backends/nvidia/include" -exec rm -rf {} + 2>/dev/null || true && \
    find "$CLEANUP_ROOT" -type d -path "*/triton/backends/nvidia/bin" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove specific non-redistributable components (regular files and symlinks)
    find "$CLEANUP_ROOT" \( -type f -o -type l \) \( \
    -name "libcheckpoint.so*" -o \
    -name "libnvperf_host.so*" -o \
    -name "libnvperf_target.so*" -o \
    -name "libpcsamplingutil.so*" -o \
    -name "libcusolverMg.so*" \
    \) -delete 2>/dev/null || true

FROM xpu-base AS xpu-clean

RUN set -eu; \
    CLEANUP_ROOT="$("/application/backend/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    [ -n "$CLEANUP_ROOT" ] && [ -d "$CLEANUP_ROOT" ]; \
    echo "Removing triton nvidia backend from ${CLEANUP_ROOT}..." && \
    # Remove triton nvidia backend
    find "$CLEANUP_ROOT" -type d -path "*/triton/backends/nvidia" -prune -exec rm -rf {} + 2>/dev/null || true

######################
# Server (CPU version)
######################
FROM runtime-base AS geti-cpu

ARG EXCLUDE_AGPL_MODELS

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=cpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

# Replace pyav
RUN uv pip install --no-deps --force-reinstall /dist/av-*.whl && \
    python -c "import av; print('PyAV:', av.__version__, av.library_versions)"

# App code + UI assets last – source-only changes only invalidate this layer
COPY --link --chown=${UID}:${UID} application/backend/app /application/backend/app
COPY --link --chown=${UID}:${UID} --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

# Optionally strip the model manifests of AGPL-3.0 licensed models
RUN if [ "$EXCLUDE_AGPL_MODELS" = "1" ]; then \
        grep -rl '^license: AGPL-3.0$' /application/backend/app/supported_models/manifests | xargs -r rm -v; \
    fi

######################
# Server (CUDA version)
######################
FROM runtime-base AS geti-cuda

ARG EXCLUDE_AGPL_MODELS

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=cuda-clean --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

# Replace pyav
RUN uv pip install --no-deps --force-reinstall /dist/av-*.whl && \
    python -c "import av; print('PyAV:', av.__version__, av.library_versions)"

# App code + UI assets last – source-only changes only invalidate this layer
COPY --link --chown=${UID}:${UID} application/backend/app /application/backend/app
COPY --link --chown=${UID}:${UID} --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

# Optionally strip the model manifests of AGPL-3.0 licensed models
RUN if [ "$EXCLUDE_AGPL_MODELS" = "1" ]; then \
        grep -rl '^license: AGPL-3.0$' /application/backend/app/supported_models/manifests | xargs -r rm -v; \
    fi

#######################
## Server (XPU version)
#######################
FROM runtime-base AS geti-xpu

ARG EXCLUDE_AGPL_MODELS

USER root

# XPU / Intel stack
# Includes Intel graphics drivers (steps from https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN getent group render_host >/dev/null || groupadd -g 992 render_host && usermod -aG render_host "${USER_NAME}"

RUN echo "deb [arch=amd64 trusted=yes] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/kobuk-intel.list

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update  \
    && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 libze-dev intel-metrics-discovery intel-opencl-icd clinfo intel-gsc intel-ocloc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo

USER ${USER_NAME}

COPY --from=xpu-clean --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

# Replace pyav
RUN uv pip install --no-deps --force-reinstall /dist/av-*.whl && \
    python -c "import av; print('PyAV:', av.__version__, av.library_versions)"

# App code + UI assets last – source-only changes only invalidate this layer
COPY --link --chown=${UID}:${UID} application/backend/app /application/backend/app
COPY --link --chown=${UID}:${UID} --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

# Optionally strip the model manifests of AGPL-3.0 licensed models
RUN if [ "$EXCLUDE_AGPL_MODELS" = "1" ]; then \
        grep -rl '^license: AGPL-3.0$' /application/backend/app/supported_models/manifests | xargs -r rm -v; \
    fi