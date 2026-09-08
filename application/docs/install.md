# Installing Intel Geti™

Geti™ is a full-stack web application to build and deploy computer vision AI models, powered by the
[getitune](../../library) library. This guide covers all the supported ways to install and run Geti™ so you
can choose the method that best fits your workflow.

## System requirements

Before you begin, make sure your machine meets the following requirements:

| Component | Requirement                            |
| --------- | -------------------------------------- |
| CPU       | 8 threads                              |
| RAM       | 16 GB                                  |
| Disk      | 40 GB free                             |
| GPU       | Optional - Intel® XPU or NVIDIA® GPU |

> [!TIP]
> For larger models, it is recommended to have a GPU (Intel® or NVIDIA®) and additional memory.

> [!NOTE]
> NVIDIA® GPUs with compute capability below 7.5 (e.g. Volta, Maxwell, Pascal) require a [manual patch](#i-have-an-old-nvidia-gpu-does-geti-support-it) to work.

## Installation methods

There are several ways to run Geti™, choose the one that best fits your workflow:

- [**Windows app**](#windows-app) - install as a native desktop application.
- [**Docker**](#run-with-docker) - download and run one of the pre-built Docker images, or build one yourself.
- [**Install script**](#install-script) - download and run a script that builds and configures Geti™ automatically.
- [**Run from source (for development)**](#run-from-source-for-development) - run the server and the UI as standalone components.

If you encounter problems during the installation, check the [Troubleshooting](#troubleshooting) section: it covers several common error scenarios, and it also explains where to find the logs. If your problem persists, please create a Github issue.

## Windows app

Installing Geti™ as a Windows app is the simplest way to run it on Windows.

1. Download the Windows Installer suitable for your hardware (prebuilt packages for Intel® XPU, NVIDIA® CUDA, and CPU-only environments):
   - [CPU-only](https://storage.geti.intel.com/geti/packages/3.1.0/geti-cpu-3.1.0.msix)
   - [Intel® XPU](https://storage.geti.intel.com/geti/packages/3.1.0/geti-xpu-3.1.0.msix)
   - [NVIDIA® CUDA](https://storage.geti.intel.com/geti/packages/3.1.0/geti-cuda-3.1.0.msix)
2. Double-click the `.msix` package and click **Install** in the Windows installer dialog.
3. Launch Geti™ from the **Start** menu.

If Windows shows a security prompt, verify that the package is from the official Geti™ release before continuing.

![Launch Geti™ from Start menu](media/geti-task-bar.jpg)

> [!IMPORTANT]
> Windows apps do not include Ultralytics models due to AGPL licensing constraints. If you need them,
> please use the [install script](#install-script) or [build a Docker image from source](#option-2-build-the-image)
> as described below.

## Run with Docker

The easiest and most portable way to run Geti™ is through Docker. Pre-built images are published for Intel® XPU
and NVIDIA® CUDA platforms; you can also build your own image from source.

### Prerequisites (on the host system)

- Docker v29+ [[Docs]](https://docs.docker.com/)
- (Optional, recommended) Just v1.46+ [[Docs]](https://just.systems/)
- (Only for Intel® XPU) the latest driver suitable with your HW [[Docs]](https://www.intel.com/content/www/us/en/download-center/home.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[Docs]](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Ubuntu 24+ or WSL2 with Ubuntu 24+ [[Docs]](https://ubuntu.com/#download-ubuntu)

### (Option 1) Download the image

Choose the most suitable image for your system:

```bash
# Choose the right image based on your hardware. Available options:
#  - xpu: best choice if you have a modern Intel® CPU or GPU
#  - cuda: recommended if you have a CUDA-enabled platform
#  - cpu: most lightweight option (CPU-only) if you don't have a compatible GPU
docker pull ghcr.io/open-edge-platform/geti-xpu
```

Retag the pulled image as `geti-{cpu,xpu,cuda}:latest` for use with `just run-image`:

```bash
docker tag ghcr.io/open-edge-platform/geti-xpu:latest geti-xpu:latest
```

> [!IMPORTANT]
> Pre-built container images do not include Ultralytics models due to AGPL licensing constraints. If you need them,
> [build the image from source](#option-2-build-the-image) as described below.

### (Option 2) Build the image

Geti™ Docker images can be built from source using the [`Dockerfile`](../docker/Dockerfile) in the `application`
directory. This is useful if you want to customize the application or include Ultralytics models. The instructions
below use `just` to simplify the build process, but you can also build the image manually with `docker build` if you prefer.

> [!TIP]
> The `develop` branch contains the latest, potentially unstable, changes. To build a specific, stable release
> instead, check out the corresponding tag (e.g. `app/v3.1.0`) before building:
>
> ```bash
> git checkout app/v3.1.0
> ```

From the `application` directory:

```bash
# Build for Intel® XPU (recommended)
just build-image --accelerator xpu
```

The above command builds an image optimized for modern Intel® hardware. If you have an Intel® GPU (discrete or integrated),
this is the recommended configuration for best performance. Alternatively, you can build with support for NVIDIA GPUs
(`--accelerator cuda`) or with CPU-only support (`--accelerator cpu`).

Run `just --usage build-image` to see all build options.

### Run the image

Once you have downloaded or built the Geti™ image, use the `run-image` command to launch the application:

```bash
# Run the image built with Intel® XPU support
just run-image --accelerator xpu
```

If you built the image with a different accelerator, make sure to specify the same one when running.

For a full list of runtime options, run `just --usage run-image`.

After the container starts, you can access the Geti™ web application at `https://localhost:7860` (assuming default settings).
If your browser warns about a self-signed certificate, choose to proceed.

<details>
<summary><strong>Custom port</strong></summary>

By default, Geti™ publishes on host port `7860`. If that port is already in use on your machine, or you simply prefer a
different one, pass `--port` to `just run-image`:

```bash
just run-image --accelerator xpu --port 8080
```

The Geti™ web application is then reachable at `https://localhost:8080`.

</details>

<details>
<summary><strong>TLS certificates</strong></summary>

By default, the container generates a self-signed certificate at startup and serves over HTTPS on the configured port.
For production deployments, mount your certificate and private key into the container using `--volumes`, then point
`--certfile` and `--keyfile` to the in-container paths:

```bash
just run-image --accelerator xpu \
    --volumes "/path/to/certs:/certs:ro" \
    --certfile /certs/server.pem \
    --keyfile  /certs/server-key.pem
```

The cert directory is mounted read-only and is separate from the data volume - it is never modified by the container.

> [!NOTE]
> The self-signed certificate triggers a browser security warning. For a trusted local setup, generate a
> locally-trusted cert with [mkcert](https://github.com/FiloSottile/mkcert) and pass it the same way.

</details>

<details>
<summary><strong>Browse the app storage</strong></summary>

The Geti™ application uses a Docker volume named `geti-data` to persistently store all datasets, models, and other objects.
You can browse the contents of this volume by running a temporary container that mounts the volume and lists the files.

```shell
# List the contents of the root directory in the `geti-data` volume
docker run --rm -v geti-data:/data alpine ls -l /data

# List the model files of a specific project (replace <PROJECT_ID> with the actual ID)
docker run --rm -v geti-data:/data alpine ls -l /data/projects/<PROJECT_ID>/models

# List the media files of a specific project (replace <PROJECT_ID> with the actual ID)
docker run --rm -v geti-data:/data alpine ls -l /data/projects/<PROJECT_ID>/dataset
```

</details>

### WebRTC preview networking and TURN

Geti™ uses WebRTC for real-time inference streaming visualization in the UI. WebRTC requires the browser to establish a
direct connection to the backend's media server. Use these options when the in-app WebRTC preview does not connect under
NAT, load balancers, or restrictive firewall policies.

| Scenario                                                  | Recommended setup                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Local machine / same LAN                                  | Usually no extra setup. If needed, set a STUN server so the host advertises a reachable address. |
| Public or cloud host with dynamic public IP               | Use STUN so the host can discover and advertise its public address.                              |
| Public host behind load balancer with fixed public IP/DNS | Configure the advertised public endpoint in your runtime setup.                                  |
| Restrictive firewall (UDP blocked, only TCP 443 allowed)  | Run a TURN relay and start Geti™ with TURN enabled.                                             |

<details>
<summary><strong>Local machine or local network</strong></summary>

On localhost this often works without changes. For LAN clients, run with STUN:

```bash
just run-image --accelerator xpu --stun stun:stun.l.google.com:19302
```

</details>

<details>
<summary><strong>Dynamic public IP (cloud or ephemeral hosts)</strong></summary>

Use STUN-based discovery:

```bash
just run-image --accelerator xpu --stun stun:stun.l.google.com:19302
```

</details>

<details>
<summary><strong>Fixed public IP or DNS</strong></summary>

If your deployment has a stable public endpoint, set that endpoint in your runtime configuration so clients do not
receive an internal/private address.

</details>

<details>
<summary><strong>Restrictive firewall: TURN relay</strong></summary>

When UDP media ports are blocked, relay media through TURN over TCP/443.

1. Start TURN relay:

   ```bash
   just run-coturn
   ```

2. Start Geti™ with TURN enabled:

   ```bash
   just run-image --accelerator xpu --coturn
   ```

3. For custom TURN endpoint settings:

   ```bash
   just run-image --accelerator xpu --coturn --coturn-host <public_ip_or_dns> --coturn-port 443
   ```

4. Stop TURN relay:

   ```bash
   just stop-coturn
   ```

Notes:

- TURN adds relay overhead. Prefer direct WebRTC when your network allows it.
- The default TURN recipe is suitable for development/validation. For production, use short-lived credentials and hardened TURN settings.
- If you run Docker directly (without `just`), publish the WebRTC UDP media port range and keep it reachable.

</details>

## Run from source (for development)

For development purposes, you can run the Geti™ server and UI as standalone components without Docker.

### Prerequisites

- Just v1.46+ [[Docs]](https://just.systems/)
- (Only for Intel® XPU) the latest driver suitable with your HW [[Docs]](https://www.intel.com/content/www/us/en/download-center/home.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[Docs]](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Node.js v24.2+ [[Docs]](https://nodejs.org/en/docs)
- Ubuntu 24+ or WSL2 with Ubuntu 24+ [[Docs]](https://ubuntu.com/#download-ubuntu)

> [!TIP]
> The `develop` branch contains the latest, potentially unstable, changes. To run a specific, stable release
> instead, check out the corresponding tag (e.g. `app/v3.1.0`) before proceeding:
>
> ```bash
> git checkout app/v3.1.0
> ```

### Run the server

To run the server, use the `run-server` command after initializing the environment with `venv`:

```bash
# From the repo root
cd application/backend

# Initialize the environment with the appropriate accelerator support (cpu, xpu, or cuda)
just venv --accelerator xpu

# Run the server
just run-server
```

Run `just --usage run-server` for a full list of options for running the server. Notably, by passing the option
`--setup-demo`, the application will be pre-populated with demo data, including sample datasets and pre-trained models.

### Run the UI

After running the server, build and launch the UI in a separate terminal:

```bash
# From the repo root
cd application/ui

# Install dependencies and build
npm install
npm run build

# Start the UI
npm run start
```

After the UI starts, you can access the Geti™ web application at `http://localhost:3000` (assuming default settings).

## Install script

The quickest way to build Geti™ from source is the install script. It downloads the source code, automatically detects
your hardware (Intel® XPU, NVIDIA® CUDA, or CPU-only), installs the required build tools, builds the backend and UI, and
registers a `geti` command you can use to launch the application.

The installer sets up its own copy of `uv`, Node.js and npm under `.build/`, then builds the backend and UI and starts
the app. The first build downloads several GB of packages (PyTorch, OpenVINO, …) and can take a while — progress is shown
for each step. Re-running the installer reuses the cached tools and dependencies, so only the first build is slow.

Installing from source also enables native Ultralytics YOLO26 models — the latest NMS-free, edge-optimized models
(Nano / Small / Medium) for object detection and instance segmentation. The integration covers the full model lifecycle:
training, inference, quantization, and OpenVINO™ model export.

### Prerequisites

- Ubuntu 24+ or WSL2 with Ubuntu 24+ [[Docs]](https://ubuntu.com/#download-ubuntu)
- `git` (required on all platforms) and `curl` (required on Linux/WSL)
- (Only for Intel® XPU) the latest driver suitable with your HW [[Docs]](https://www.intel.com/content/www/us/en/download-center/home.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[Docs]](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

### Download and run the installer

**Linux / WSL2**

Run the command below to download the script and start the installation in one step. The `--yes` flag runs it
non-interactively, accepting the default prompts:

```bash
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/geti/develop/install.sh | bash -s -- --yes
```

Prefer to inspect the script before running it? Download it first, then execute it (this also lets you answer the prompts interactively):

```bash
# Download the installer
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/geti/develop/install.sh -o install.sh

# (Optional) Review the script, then run it
bash install.sh
```

You can forward flags through the pipe with `bash -s --`: `-v`/`--verbose` (stream full output), `-y`/`--yes`
(non-interactive), `-w`/`--work-dir <path>` (custom install directory, default `./geti`):

```bash
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/geti/develop/install.sh | bash -s -- --yes --work-dir ~/geti
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/open-edge-platform/geti/develop/install.ps1 | iex
```

To pass parameters — `-Verbose` (stream full output), `-Yes`/`-y` (non-interactive), `-WorkDir <path>`/`-w` (custom
install directory, default `.\geti`) — run the downloaded script as a script block instead:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/open-edge-platform/geti/develop/install.ps1))) -Yes -WorkDir C:\geti
```

If your execution policy blocks remote scripts, download first and run it explicitly (Bypass applies only to this
process and does not change your machine policy):

```powershell
curl.exe -L https://raw.githubusercontent.com/open-edge-platform/geti/develop/install.ps1 -o install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

If a build step fails, re-run with `--verbose` (Linux) or `-Verbose` (Windows), or inspect the log at
`<work-dir>/.build/.install.log`.

### Launch Geti™

When the script finishes, it can add a `geti` command to your shell profile (on Windows, you’ll be prompted). Reload your shell profile, then start the application:

```bash
# Reload your shell profile (use ~/.zshrc if you use zsh)
source ~/.bashrc  # or: source ~/.zshrc

# Start Geti™ (optionally set HOST and PORT)
HOST=0.0.0.0 PORT=8080 geti

# Start Geti™ with STUN enabled
STUN_SERVER=stun:stun.l.google.com:19302 HOST=0.0.0.0 PORT=8080 geti

# Start Geti™ using the active COTURN server (ensuring the IP and port match the running instance)
COTURN_HOST=10.123.246.153 COTURN_PORT=443 HOST=0.0.0.0 PORT=8080 geti
```

Then open the Geti™ web application at `https://localhost:8080`. If your browser warns about a self-signed certificate,
choose to proceed.

## Advanced

<details>
<summary><strong>Air-gapped setup (pretrained weights cache)</strong></summary>

When deploying Geti™ in an air-gapped or restricted network environment, the application cannot automatically download
pretrained model weights from remote repositories. You must manually preconfigure and populate the pretrained weights
cache before running the application.

#### 1. Locate the download links

To find the exact URLs for the pretrained weights required by a specific model:

- Navigate to the model manifest files bundled with the Geti™ application backend (e.g., `application/backend/app/supported_models/manifests/classification/vit_tiny.yaml`).
- Look for the `pretrained_weights.url` attribute within the manifest file to retrieve the direct download link.
- Note the corresponding `pretrained_weights.sha_sum` attribute - you'll use it to verify the integrity of the downloaded file before placing it in the cache.
- Note the `pretrained_weights.cache_filename` attribute, if present. This is the filename to use when placing the weights in the cache. If unspecified, use the same basename specified in the URL.

#### 2. Verify file integrity

Before transferring the downloaded weights file into the air-gapped environment, verify it hasn't been tampered with or
corrupted by comparing its SHA-256 checksum against the `sha_sum` value from the manifest:

```shell
# Compute the SHA-256 checksum of the downloaded file
shasum -a 256 /path/to/local/downloaded/<WEIGHTS_FILE_NAME>

# Compare the output against the `pretrained_weights.sha_sum` value in the model manifest.
# The two values must match exactly before proceeding.
```

> [!CAUTION]
> Do not place a weights file into the cache if its checksum doesn't match the manifest's sha_sum. Geti™ performs this
> verification automatically when loading cached weights, and a mismatch will cause the file to be treated as invalid -
> but validating it upfront avoids wasting time transferring a corrupted or tampered file into an air-gapped environment.

#### 3. Populate the pretrained weights cache

Once you have downloaded and verified the integrity of the required weights file on an internet-connected machine,
transfer it to your air-gapped environment and place it in the application data directory.

Because Geti™ stores its application data inside the `geti-data` Docker volume, you can use a temporary container to
inject the downloaded file into the proper cache directory:

```shell
# Copy the downloaded weights file into the application's pretrained cache folder
docker run --rm -i -v geti-data:/data alpine sh -c "mkdir -p /data/pretrained_weights/<TASK_TYPE> && cat > /data/pretrained_weights/<TASK_TYPE>/<WEIGHTS_FILE_NAME>" < /path/to/local/downloaded/<WEIGHTS_FILE_NAME>

# Verify that the file is correctly placed
docker run --rm -v geti-data:/data alpine ls -l /data/pretrained_weights/<TASK_TYPE>
```

> [!NOTE]
> Replace `<TASK_TYPE>` with the corresponding model task type (e.g., `classification`, `detection`, or `instance_segmentation`).
> Replace `<WEIGHTS_FILE_NAME>` with the exact filename retrieved from the manifest.
> Replace `/path/to/local/downloaded/` with the path to the file on your host machine.

</details>

## Upgrading

- **Upgrading an existing Geti™ v3 installation to a newer release?** See the [Upgrade guide](./upgrade.md) for how to
  move to a newer version (Docker or Windows MSIX) while preserving your projects, datasets and models, with automatic
  rollback if a migration fails.
- **Upgrading from Geti™ v2 to v3?** The two versions have different architectures and storage layouts. Follow the
  dedicated [migration guidance](https://docs.geti.intel.com/docs/user-guide/getting-started/installation/migration-from-geti-2x)
  in the Geti™ documentation.

## Uninstalling

- **Docker:** Stop and remove the running container, then remove the pulled image. To also delete all stored data,
  remove the `geti-data` volume (this is irreversible):

  ```bash
  # Stop and remove the container (replace <container> with the container name or ID)
  docker stop <container> && docker rm <container>

  # (Optional) Remove the image (instead of xpu, use `geti-cuda` or `geti-cpu` if you pulled those images)
  docker rmi ghcr.io/open-edge-platform/geti-xpu

  # (Optional, irreversible) Delete all stored datasets, models, and logs
  docker volume rm geti-data geti-logs
  ```

- **MSIX (Windows app):** Uninstall the package from the **Start** menu (right-click **Geti™ > Uninstall**) or via
  **Settings > Apps > Installed apps > Geti™ > Uninstall**.

<a id="troubleshooting"></a>

## FAQ & Troubleshooting

<a id="troubleshooting-logs"></a>

<details>
<summary><strong>How can I view the Geti logs?</strong></summary>

The location depends on how Geti™ is installed:

- Windows app (MSIX): `%LOCALAPPDATA%\Intel\Geti` folder.
- Docker container: `geti-logs` Docker volume.

In the Docker case, you can view these logs by running
a temporary container that mounts the volume and prints the log files to the console. The following examples use `jq` to format the JSON logs; install `jq` on the host or omit the `| jq -r '.text'` part to see the
raw JSON output.

**Application logs:**

```bash
# Print the logs of the application container to the console
docker run --rm -v geti-logs:/logs alpine cat /logs/app.log | jq -r '.text'

# Or save the logs to a file for easier browsing
docker run --rm -v geti-logs:/logs alpine cat /logs/app.log | jq -r '.text' > geti-logs.txt
```

**Job logs:**

```bash
# List the available job logs
docker run --rm -v geti-logs:/logs alpine ls -l /logs/jobs

# Print the logs of a specific job to the console
docker run --rm -v geti-logs:/logs alpine cat /logs/jobs/<job_type>-<job_id>.log | jq -r '.text'
```

**Logs of other worker processes:**

```bash
# Print the logs of the inference pipeline stream loader
docker run --rm -v geti-logs:/logs alpine cat /logs/workers/streamloader.log | jq -r '.text'

# Print the logs of the inference worker
docker run --rm -v geti-logs:/logs alpine cat /logs/workers/inference.log | jq -r '.text'
```

</details>

<details>
<summary><strong>I have an old Nvidia GPU, does Geti support it?</strong></summary>

<a id="i-have-an-old-nvidia-gpu-does-geti-support-it"></a>

Geti's `cuda` builds use PyTorch wheels compiled against CUDA 13.0, which dropped support for NVIDIA GPU architectures older than Turing (compute capability < 7.5), covering Volta, Maxwell, and Pascal cards. If your GPU falls in this range, you must patch the source to build against CUDA 12.6 instead, which still supports
compute capability >= 5.0.

- **Install script:** patch `pyproject.toml` before running the installer:

  ```bash
  git clone https://github.com/open-edge-platform/geti.git
  cd geti/application
  just patch-for-legacy-gpu-support
  # Then run install.sh / install.ps1 as usual, from the repo root
  ```

- **Docker (build from source):** run the patch recipe from the `application` directory, then build the image as usual:

  ```bash
  cd application
  just patch-for-legacy-gpu-support
  just build-image --accelerator cuda
  ```

- **Run from source (development):** run the patch recipe from the `application` directory before initializing the backend environment:

  ```bash
  cd application
  just patch-for-legacy-gpu-support
  cd backend
  just venv --accelerator cuda
  ```

> [!NOTE]
> The Windows app (MSIX) is prebuilt and cannot be patched this way; it requires the newer PyTorch/CUDA wheels.

</details>

<details>
<summary><strong>Just fails with "unknown start of token" or "unknown keyword"</strong></summary>

This error means the installed version of `just` is too old to support the `[arg(...)]` attribute syntax used in
this repository's `Justfile`s. Update to the latest release by following the
[Just installation documentation](https://just.systems/man/en/packages.html), then re-run the command.

</details>

<details>
<summary><strong>Docker fails with "unknown flag: --build-arg"</strong></summary>

This means the Docker CLI is missing the `buildx` plugin, which provides the modern build backend used by our
build recipes. Follow the [Docker documentation](https://docs.docker.com/engine/install/) to install Docker
cleanly (e.g. via the official `docker-ce`/`docker-buildx-plugin` packages or Docker Desktop) rather than a
partial or outdated installation, then retry the build.

</details>

<details>
<summary><strong>Docker or Just commands fail with "permission denied"</strong></summary>

This typically happens when your user isn't allowed to talk to the Docker daemon socket. Follow the
[Docker post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) to add your user to
the `docker` group, then log out and back in (or restart your session) for the change to take effect.

</details>

<details>
<summary><strong>WSL2: WebRTC preview connects but video freezes after a few frames</strong></summary>

When the backend runs in **WSL2** and the browser runs on the **Windows host**, the WebRTC preview may show
the first few frames and then freeze. Typical symptoms: the browser's `framesReceived` stops increasing while
`bytesReceived` keeps growing and `packetsLost` climbs steadily. This means bytes reach the browser but complete
video frames can never be reassembled — the classic signature of large RTP/UDP packets being dropped by WSL2's
virtual NIC (an MTU/NAT issue), **not** an ICE candidate problem. Because the byte path itself is broken,
`--webrtc-advertise-ip`, `--stun`, and `--coturn` all fail identically.

Fixes, in order of preference:

1. **Enable WSL2 mirrored networking (recommended).** This makes WSL share the Windows host network stack and
   removes the virtual NAT/MTU boundary entirely. Add the following to `%UserProfile%\.wslconfig` on Windows and
   restart WSL (`wsl --shutdown`):

   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. **Lower the WSL2 interface MTU** so RTP packets fit without fragmentation, then reconnect the stream:

   ```bash
   sudo ip link set dev eth0 mtu 1400
   ```

   If this helps, make it persistent (e.g. via a WSL boot command in `/etc/wsl.conf`).

3. **Run the backend and browser on the same side** (both inside WSL, or use the Docker image on the Windows host)
   to avoid crossing the WSL2 network boundary altogether.

</details>

<br>

Still can't find a solution for your issue? Feel free to [open a Github issue](https://github.com/open-edge-platform/geti/issues) and the Geti™ team will help you.

## Notes

> [!NOTE]
> Ultralytics YOLO models are distributed under the AGPL-3.0 license, an OSI approved license ideal for open-source
> research, academic, and personal projects. For commercial use, enhanced support, and tailored licensing terms, please
> explore flexible Ultralytics licensing options at https://www.ultralytics.com/license.
