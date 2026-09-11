#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "ERROR: Installation failed at line $1 (exit code $exit_code)."
        if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then
            echo "Check $LOG_FILE for details."
        fi
        echo "Re-run with --verbose for more details."
    fi
    exit $exit_code
}

trap 'cleanup $LINENO' ERR
trap 'echo ""; echo "Installation interrupted."; exit 130' INT TERM

GIT_URL="https://github.com/open-edge-platform/geti.git"
# GIT_BRANCH can be overridden via the GIT_BRANCH environment variable or the
# --git-branch flag (for testing purposes).
GIT_BRANCH="${GIT_BRANCH:-app/v3.2.0}"

# Exit code the backend uses for a fatal, non-restartable migration failure
# (see application/backend/app/lifecycle.py:MIGRATION_FATAL_EXIT_CODE). It lets
# the upgrade path distinguish "the data could not be migrated" from an ordinary
# crash so it can trigger a rollback.
readonly MIGRATION_FATAL_EXIT_CODE=3

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install OR upgrade the Intel Geti application and its dependencies.

Re-running this script on an existing installation upgrades it in place: the
source is updated to the target version and the backend migrates your data on
startup. Existing application data is backed up first and, if anything fails,
the previous version and data are automatically restored so the app stays
usable.

Options:
  -v, --verbose         Show detailed output from all commands
  -y, --yes             Assume yes to all prompts (non-interactive mode)
  -w, --work-dir        Set the working directory (default: \$PWD/geti)
  -u, --upgrade         Force upgrade mode even if no existing data is detected
      --no-data-backup  Skip the pre-upgrade data backup (NOT recommended;
                        relies solely on the backend's own DB rollback)
      --keep-backup     Keep the pre-upgrade data backup after a successful upgrade
      --backup-dir DIR  Directory for pre-upgrade data backups
                        (default: <work-dir>/.geti-upgrade-backups)
      --health-timeout N  Seconds to wait for the upgraded app to become healthy
                        before rolling back (default: 300)
      --git-branch REF  Override the git branch/tag to install (for testing)
  -h, --help            Show this help message and exit
EOF
}

parse_args() {
    VERBOSE=""
    ASSUME_YES=""
    WORK_DIR="$(pwd)/geti"
    FORCE_UPGRADE=""
    DATA_BACKUP=1
    KEEP_BACKUP=""
    BACKUP_DIR=""
    HEALTH_TIMEOUT=300

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -y|--yes)
                ASSUME_YES=1
                shift
                ;;
            -w|--work-dir)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --work-dir requires a path argument."
                    exit 1
                fi
                WORK_DIR="$2"
                shift 2
                ;;
            -u|--upgrade)
                FORCE_UPGRADE=1
                shift
                ;;
            --no-data-backup)
                DATA_BACKUP=""
                shift
                ;;
            --keep-backup)
                KEEP_BACKUP=1
                shift
                ;;
            --backup-dir)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --backup-dir requires a path argument."
                    exit 1
                fi
                BACKUP_DIR="$2"
                shift 2
                ;;
            --health-timeout)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --health-timeout requires a number argument."
                    exit 1
                fi
                HEALTH_TIMEOUT="$2"
                shift 2
                ;;
            --git-branch)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --git-branch requires a ref argument."
                    exit 1
                fi
                GIT_BRANCH="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Error: unknown option '$1'"
                usage
                exit 1
                ;;
        esac
    done

    BUILD_TOOLS_DIR="$WORK_DIR/.build"
    UV_DIR="$BUILD_TOOLS_DIR/uv"
    NVM_DIR="$BUILD_TOOLS_DIR/nvm"
    LOG_FILE="$BUILD_TOOLS_DIR/.install.log"

    # Upgrade-related derived paths.
    DATA_PATH="$WORK_DIR/application/backend/data"
    if [[ -z "$BACKUP_DIR" ]]; then
        BACKUP_DIR="$WORK_DIR/.geti-upgrade-backups"
    fi
    TIMESTAMP="$(date -u +%Y%m%d%H%M%S)"
    DATA_BACKUP_FILE="$BACKUP_DIR/geti-data-${TIMESTAMP}.tar.gz"
    # Populated during an upgrade so the rollback path knows what to restore.
    PREVIOUS_SHA=""
    DATA_BACKED_UP=""
}

confirm() {
    local prompt="$1"
    if [ -n "${ASSUME_YES:-}" ]; then
        return 0
    fi
    local response
    if [ -t 0 ]; then
        read -rp "$prompt [Y/n]: " response
    elif [ -e /dev/tty ]; then
        read -rp "$prompt [Y/n]: " response </dev/tty
    else
        echo "Error: confirmation required but no terminal is available."
        echo "Re-run with -y/--yes to skip prompts in non-interactive mode."
        exit 1
    fi
    if [[ "${response,,}" =~ ^n(o)?$ ]]; then
        return 1
    fi
    return 0
}

run_cmd() {
    if [ -n "${VERBOSE:-}" ]; then
        "$@"
    else
        "$@" >>"$LOG_FILE" 2>&1
    fi
}

# Echo a timestamped message to the console and append it to the log file. Used
# by the upgrade path so the sequence of upgrade/rollback actions is captured
# for troubleshooting.
log() {
    local line
    line="$(date -u +%H:%M:%S) $*"
    echo "$line"
    if [ -n "${LOG_FILE:-}" ]; then
        echo "$line" >>"$LOG_FILE" 2>/dev/null || true
    fi
}

run_cmd_spinner() {
    # Run a long command quietly (output to the log file) while showing an
    # animated spinner, so the step never looks frozen. In verbose mode the
    # full output is streamed instead.
    local activity="$1"
    shift

    if [ -n "${VERBOSE:-}" ]; then
        echo "${activity}..."
        "$@"
        return
    fi

    "$@" >>"$LOG_FILE" 2>&1 &
    local pid=$!
    local spin='|/-\'
    local i=0
    if [ -t 2 ]; then
        while kill -0 "$pid" 2>/dev/null; do
            i=$(( (i + 1) % 4 ))
            printf "\r%s... %s" "$activity" "${spin:$i:1}" >&2
            sleep 0.2
        done
    fi

    local rc=0
    wait "$pid" || rc=$?
    if [ "$rc" -eq 0 ]; then
        printf "\r%s... done   \n" "$activity" >&2
    else
        printf "\r%s... failed\n" "$activity" >&2
        return "$rc"
    fi
}

get_required_uv_version() {
    local version
    version=$(grep -A 3 '\[tool\.uv\]' "$WORK_DIR/application/backend/pyproject.toml" \
        | grep 'required-version' \
        | sed -E 's/.*"[^0-9]*([0-9]+\.[0-9]+\.[0-9]+).*/\1/' || true)
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: could not parse uv version from pyproject.toml" >&2
        return 1
    fi
    echo "$version"
}

get_required_node_version() {
    local version
    version=$(grep '"node"' "$WORK_DIR/application/ui/package.json" \
        | sed -E 's/.*">=v?([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: could not parse node version from package.json" >&2
        return 1
    fi
    echo "$version"
}

get_required_npm_version() {
    local version
    version=$(grep '"npm"' "$WORK_DIR/application/ui/package.json" \
        | sed -E 's/.*">=([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: could not parse npm version from package.json" >&2
        return 1
    fi
    echo "$version"
}

install_uv() {
    local uv_version
    uv_version=$(get_required_uv_version)

    if [ -x "$UV_DIR/uv" ]; then
        local installed_version
        installed_version=$("$UV_DIR/uv" --version | awk '{print $2}')
        if [ "$installed_version" = "$uv_version" ]; then
            echo "uv $uv_version found in $UV_DIR"
            return 0
        else
            echo "uv version mismatch: installed=$installed_version, required=$uv_version. Reinstalling..."
        fi
    fi

    echo "Installing uv $uv_version to: $UV_DIR"
    if ! confirm "Would you like to install uv now?"; then
        echo "uv installation skipped. Cannot continue without uv."
        exit 1
    fi

    if [ ! -d "$UV_DIR" ]; then
        mkdir -p "$UV_DIR"
    fi

    run_cmd_spinner "Downloading and installing uv $uv_version" bash -c "curl --proto '=https' --tlsv1.2 -LsSf 'https://github.com/astral-sh/uv/releases/download/${uv_version}/uv-installer.sh' | env UV_INSTALL_DIR='$UV_DIR' sh"
    echo "uv installation complete."
}

install_nvm() {
    export NVM_DIR

    if [ -s "$NVM_DIR/nvm.sh" ]; then
        source "$NVM_DIR/nvm.sh"
        echo "nvm found in $NVM_DIR."
        return 0
    fi

    echo "Installing nvm to: $NVM_DIR"
    if ! confirm "Would you like to install nvm now?"; then
        echo "nvm installation skipped. Cannot continue without nvm."
        exit 1
    fi

    if [ ! -d "$NVM_DIR" ]; then
        mkdir -p "$NVM_DIR"
    fi

    run_cmd_spinner "Downloading and installing nvm" bash -c "curl -sS -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash"
    source "$NVM_DIR/nvm.sh"
    echo "nvm installation complete."
}

install_npm() {
    local required_node_version required_npm_version
    required_node_version=$(get_required_node_version)
    required_npm_version=$(get_required_npm_version)

    NPM_BIN="$NVM_DIR/versions/node/v${required_node_version}/bin/npm"
    local node_bin installed_npm_version
    node_bin="$(dirname "$NPM_BIN")"

    # npm is a script with a `#!/usr/bin/env node` shebang, so every npm invocation
    # needs `node` on PATH. `nvm install` cannot provide it: run_cmd_spinner executes
    # it in a backgrounded subshell, so the PATH nvm exports is discarded when that
    # subshell exits. Sourcing nvm.sh only helps on later runs, once a default alias
    # exists -- which is why a first-time install failed with
    # "env: 'node': No such file or directory" but a re-run succeeded. Put the node
    # bin directory on PATH here instead of relying on nvm.
    case ":$PATH:" in
        *":$node_bin:"*) ;;
        *) export PATH="$node_bin:$PATH" ;;
    esac

    if [ -x "$node_bin/node" ] && [ -x "$NPM_BIN" ]; then
        installed_npm_version=$("$NPM_BIN" --version)
        if [ "$(printf '%s\n' "$required_npm_version" "$installed_npm_version" | sort -V | head -n1)" = "$required_npm_version" ]; then
            echo "node $required_node_version and npm $installed_npm_version found in $node_bin."
            return 0
        fi

        echo "npm version too old: installed=$installed_npm_version, required>=$required_npm_version. Upgrading..."
        run_cmd "$NPM_BIN" install -g "npm@$required_npm_version"
        return 0
    fi

    echo "Required node $required_node_version not found in $NVM_DIR. Installing..."
    run_cmd_spinner "Downloading and installing node $required_node_version" nvm install "$required_node_version"

    if [ ! -x "$node_bin/node" ] || [ ! -x "$NPM_BIN" ]; then
        echo "Error: node $required_node_version was not installed into $node_bin." >&2
        echo "See $LOG_FILE for the nvm output." >&2
        return 1
    fi

    installed_npm_version=$("$NPM_BIN" --version)
    if [ "$(printf '%s\n' "$required_npm_version" "$installed_npm_version" | sort -V | head -n1)" != "$required_npm_version" ]; then
        run_cmd "$NPM_BIN" install -g "npm@$required_npm_version"
    fi

    echo "node/npm installation complete: $node_bin"
}


detect_nvidia_gpus() {
    local gpu_count=0

    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)
        if [ "$gpu_count" -gt 0 ]; then
            echo "Detected $gpu_count NVIDIA GPU(s) via nvidia-smi:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
            return 0
        fi
    fi

    if [ -d /proc/driver/nvidia/gpus ]; then
        gpu_count=$(ls /proc/driver/nvidia/gpus 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo "Detected $gpu_count NVIDIA GPU(s) via /proc/driver/nvidia/gpus"
            return 0
        fi
    fi

    if command -v lspci &>/dev/null; then
        local gpus
        gpus=$(lspci | grep -i 'nvidia' | grep -i 'vga\|3d\|display' || true)
        if [ -n "$gpus" ]; then
            gpu_count=$(echo "$gpus" | wc -l)
            echo "Detected $gpu_count NVIDIA GPU(s) via lspci:"
            echo "$gpus"
            return 0
        fi
    fi

    echo "No NVIDIA GPUs detected."
    return 1
}

detect_intel_gpus() {
    local gpu_count=0

    if command -v xpu-smi &>/dev/null; then
        gpu_count=$(xpu-smi discovery 2>/dev/null | grep -c 'Device ID' || true)
        if [ "$gpu_count" -gt 0 ]; then
            echo "Detected $gpu_count Intel GPU(s) via xpu-smi:"
            xpu-smi discovery 2>/dev/null
            return 0
        fi
    fi

    if command -v sycl-ls &>/dev/null; then
        local intel_devs
        intel_devs=$(sycl-ls 2>/dev/null | grep -i 'intel' || true)
        if [ -n "$intel_devs" ]; then
            gpu_count=$(echo "$intel_devs" | wc -l)
            echo "Detected Intel GPU(s) via sycl-ls:"
            echo "$intel_devs"
            return 0
        fi
    fi

    if command -v lspci &>/dev/null; then
        local gpus
        gpus=$(lspci | grep -i 'intel' | grep -i 'vga\|3d\|display' || true)
        if [ -n "$gpus" ]; then
            gpu_count=$(echo "$gpus" | wc -l)
            echo "Detected $gpu_count Intel GPU(s) via lspci:"
            echo "$gpus"
            return 0
        fi
    fi

    echo "No Intel GPUs detected."
    return 1
}

preflight_checks() {
    if ! command -v git &>/dev/null; then
        echo "Error: git is not installed. Please install git and try again."
        exit 1
    fi

    if ! command -v curl &>/dev/null; then
        echo "Error: curl is not installed. Please install curl and try again."
        exit 1
    fi
}

ensure_source_code() {
    if [ ! -d "$WORK_DIR" ]; then
        echo "Cloning Intel Geti repository from $GIT_URL..."
        echo "This can take several minutes depending on your connection."
        git -c advice.detachedHead=false clone --progress --branch "$GIT_BRANCH" "$GIT_URL" "$WORK_DIR"
    else
        echo "Work directory $WORK_DIR already exists, skipping clone."
        local remote_url
        remote_url=$(git -C "$WORK_DIR" remote get-url origin 2>/dev/null)
        if [ "$remote_url" != "$GIT_URL" ]; then
            echo "Error: $WORK_DIR remote origin is '$remote_url', expected '$GIT_URL'."
            echo "Remove $WORK_DIR and re-run the installer."
            exit 1
        fi
        local current_sha expected_sha
        current_sha=$(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null)
        # Fetch: try as tag first, then as branch
        git -C "$WORK_DIR" fetch origin "refs/tags/${GIT_BRANCH}:refs/tags/${GIT_BRANCH}" --force 2>/dev/null \
            || git -C "$WORK_DIR" fetch origin "$GIT_BRANCH" --tags 2>/dev/null \
            || true
        # Resolve: try as tag first, then as remote branch
        expected_sha=$(git -C "$WORK_DIR" rev-parse "refs/tags/$GIT_BRANCH" 2>/dev/null) \
            || expected_sha=$(git -C "$WORK_DIR" rev-parse "origin/$GIT_BRANCH" 2>/dev/null) \
            || true
        if [ -z "$expected_sha" ] || ! echo "$expected_sha" | grep -qE '^[0-9a-f]{40}$'; then
            echo "Error: Could not resolve ref '$GIT_BRANCH'. Ensure it exists on the remote."
            exit 1
        fi
        if [ "$current_sha" != "$expected_sha" ]; then
            echo "Switching to $GIT_BRANCH..."
            git -c advice.detachedHead=false -C "$WORK_DIR" checkout --force "$GIT_BRANCH"
        fi
    fi
}

install_build_tools() {
    install_uv
    install_nvm
    install_npm
}

detect_hardware() {
    HAS_NVIDIA_GPU=false
    HAS_INTEL_GPU=false

    if detect_nvidia_gpus; then
        HAS_NVIDIA_GPU=true
    fi

    if detect_intel_gpus; then
        HAS_INTEL_GPU=true
    fi

    if [ "$HAS_NVIDIA_GPU" = true ]; then
        ACCELERATOR="cuda"
    elif [ "$HAS_INTEL_GPU" = true ]; then
        ACCELERATOR="xpu"
    else
        ACCELERATOR="cpu"
    fi

    export ACCELERATOR
}

build_backend() {
    echo "Building Python environment using accelerator: $ACCELERATOR"
    echo "This downloads PyTorch, OpenVINO and other large packages and can take several minutes."
    cd "$WORK_DIR/application/backend"
    # uv shows its own progress meter; do not suppress it so the user gets feedback.
    "$UV_DIR/uv" sync --frozen --extra mqtt --extra "$ACCELERATOR"

    echo "Generating OpenAPI specification..."
    PYTHONPATH=. "$UV_DIR/uv" run --no-sync app/cli.py gen-api --target-path openapi.json
    cp openapi.json ../ui/src/api/openapi-spec.json
}

build_frontend() {
    cd "$WORK_DIR/application/ui"
    export npm_config_yes=true

    # Remove build artifacts and cloned workspace packages left over from a
    # previous build/version. `git checkout --force` does not touch these
    # untracked paths, and stale contents make `npm ci` fail with
    # "package.json and package-lock.json are not in sync".
    rm -rf node_modules packages dist

    run_cmd_spinner "Installing UI dependencies (this may take several minutes)" "$NPM_BIN" ci --foreground-scripts

    run_cmd_spinner "Building API client" "$NPM_BIN" run build:api

    run_cmd_spinner "Building UI (this may take several minutes)" env ASSET_PREFIX="/html" "$NPM_BIN" run build
}

deploy_frontend() {
    local html_dir="$WORK_DIR/application/backend/html"

    echo "Copying built UI to backend html directory..."
    if [ -d "$html_dir" ]; then
      rm -rf "$html_dir"
    fi
    mkdir "$html_dir"
    cp -r "$WORK_DIR/application/ui/dist/"* "$html_dir"
}

register_shell_cmd() {
    local begin_marker="# BEGIN Intel Geti"
    local end_marker="# END Intel Geti"
    local shell_profile="${HOME}/.bashrc"

    if [ -n "${ZSH_VERSION:-}" ] || [[ "$SHELL" == */zsh ]]; then
        shell_profile="${HOME}/.zshrc"
    fi

    # Remove old marker block if present (idempotent update)
    if grep -qF "$begin_marker" "$shell_profile" 2>/dev/null; then
        sed -i "/$begin_marker/,/$end_marker/d" "$shell_profile"
    fi

    {
        echo ""
        echo "$begin_marker"
        echo "function geti { (cd '$WORK_DIR/application/backend' && STATIC_FILES_DIR=html '$UV_DIR/uv' run app/main.py \"\$@\"); }"
        echo "$end_marker"
    } >> "$shell_profile"

    echo "Function 'geti' written to $shell_profile"
    echo "Run 'source $shell_profile' to activate it in the current session."
    echo "Example: HOST=0.0.0.0 PORT=8080 geti"
}

build_and_deploy() {
    install_build_tools
    detect_hardware
    build_backend
    build_frontend
    deploy_frontend
}

# ---------------------------------------------------------------------------
# Upgrade support
# ---------------------------------------------------------------------------

# Decide whether this run is a fresh install or an in-place upgrade. An upgrade
# is any run against an existing source checkout that already holds application
# data (or when the user forces it with --upgrade). A checkout with no data is
# still treated as a fresh install so first-time builds are never gated behind
# the (heavier) upgrade path.
detect_upgrade() {
    IS_UPGRADE=false
    if [ ! -d "$WORK_DIR/.git" ]; then
        # No existing checkout — nothing to upgrade.
        return
    fi
    if [ -n "${FORCE_UPGRADE:-}" ]; then
        IS_UPGRADE=true
        return
    fi
    if [ -d "$DATA_PATH" ] && [ -n "$(ls -A "$DATA_PATH" 2>/dev/null)" ]; then
        IS_UPGRADE=true
    fi
}

# Record everything needed to restore the current version: the git revision and
# a full snapshot of the application data directory.
capture_rollback_point() {
    PREVIOUS_SHA="$(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [ -n "$PREVIOUS_SHA" ]; then
        log "Recorded current source version: $PREVIOUS_SHA"
    else
        log "WARNING: could not determine current git revision; source rollback disabled."
    fi

    if [ -z "${DATA_BACKUP:-}" ]; then
        log "WARNING: --no-data-backup set; skipping pre-upgrade data backup."
        return
    fi
    if [ ! -d "$DATA_PATH" ] || [ -z "$(ls -A "$DATA_PATH" 2>/dev/null)" ]; then
        log "No existing application data found; nothing to back up."
        return
    fi

    mkdir -p "$BACKUP_DIR"
    log "Backing up application data → ${DATA_BACKUP_FILE} ..."
    log "(This may take a while and disk space proportional to your data size.)"
    run_cmd tar czf "$DATA_BACKUP_FILE" -C "$DATA_PATH" .
    DATA_BACKED_UP=1
    log "✓ Data backup created."
}

# Start the freshly built backend and wait until it reports healthy, so a failed
# data migration is caught before we consider the upgrade successful. The
# verification instance is stopped once healthy; the app is (re)started normally
# by run_app afterwards.
verify_app_start() {
    local port="${PORT:-7860}"
    log "Verifying the upgraded application starts and migrates data (up to ${HEALTH_TIMEOUT}s)..."

    (
        cd "$WORK_DIR/application/backend"
        STATIC_FILES_DIR=html "$UV_DIR/uv" run app/main.py
    ) >>"$LOG_FILE" 2>&1 &
    local app_pid=$!

    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if ! kill -0 "$app_pid" 2>/dev/null; then
            local code=0
            wait "$app_pid" || code=$?
            if [ "$code" -eq "$MIGRATION_FATAL_EXIT_CODE" ]; then
                log "✗ Backend exited with fatal migration code ${code}: data could not be migrated."
            else
                log "✗ Backend exited unexpectedly (exit code ${code}) during verification."
            fi
            return 1
        fi
        # The backend serves /health over HTTPS with a self-signed cert (-k).
        if curl -ksSf --max-time 5 "https://localhost:${port}/health" >/dev/null 2>&1; then
            log "✓ Upgraded version is healthy."
            stop_verification "$app_pid"
            return 0
        fi
        sleep 3
    done

    log "✗ Timed out waiting for the upgraded app to become healthy."
    stop_verification "$app_pid"
    return 1
}

# Recursively terminate a process and all of its descendants. The backend spawns
# uv → python → worker processes, so killing only the top PID would orphan the
# rest and keep the port bound before run_app rebinds it.
kill_tree() {
    local pid="$1"
    local child
``    # `pgrep` exits 1 when a process has no children. Under `set -e` + `set -E`
    # (errtrace) that non-zero status inside the `$(...)` subshell would fire the
    # inherited ERR trap, so swallow it with `|| true`.
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_tree "$child"
    done
    kill -TERM "$pid" 2>/dev/null || true
}

stop_verification() {
    local app_pid="$1"
    kill_tree "$app_pid"
    wait "$app_pid" 2>/dev/null || true
}

# Restore the previous version (source + data) and rebuild it so the app remains
# usable after a failed upgrade.
# $1: line number the upgrade failed on (passed by the ERR trap), if known.
upgrade_rollback() {
    # `set -E` (errtrace) makes this ERR trap fire inside command substitutions
    # and other subshells too (e.g. a `$(pgrep ...)` that exits non-zero). A
    # rollback must only ever run in the top-level shell: from a subshell its
    # `exit 1` and the ROLLBACK_IN_PROGRESS guard would be discarded when the
    # subshell ends, the parent would keep going with the trap still armed, and
    # the trap would re-fire — rebuilding the app over and over. Ignore any such
    # subshell invocation and let the real failure surface in the main shell.
    if [ "$BASHPID" != "$$" ]; then
        return
    fi
    trap - ERR
    set +e
    # The rollback itself rebuilds the app, so a second entry would restart the whole
    # build and make the installer look like it is looping. Only ever run once.
    if [ -n "${ROLLBACK_IN_PROGRESS:-}" ]; then
        log "A rollback is already in progress${1:+ (second failure at line $1)}; ignoring this failure."
        return
    fi
    ROLLBACK_IN_PROGRESS=1
    echo ""
    log "──────────────────────────────────────────────"
    if [ -n "${1:-}" ]; then
        log "Upgrade failed at line $1 of the installer. Rolling back to the previous version..."
    else
        log "Upgrade failed. Rolling back to the previous version..."
    fi

    if [ -n "${DATA_BACKED_UP:-}" ] && [ -f "$DATA_BACKUP_FILE" ]; then
        log "Restoring application data from backup..."
        rm -rf "$DATA_PATH"
        mkdir -p "$DATA_PATH"
        if tar xzf "$DATA_BACKUP_FILE" -C "$DATA_PATH" >>"$LOG_FILE" 2>&1; then
            log "✓ Data restored to its pre-upgrade state."
        else
            log "✗ Could not restore data. Your backup is preserved at ${DATA_BACKUP_FILE}."
        fi
    fi

    if [ -n "${PREVIOUS_SHA:-}" ]; then
        log "Restoring previous source version (${PREVIOUS_SHA})..."
        git -c advice.detachedHead=false -C "$WORK_DIR" checkout --force "$PREVIOUS_SHA" >>"$LOG_FILE" 2>&1 || true
        log "Rebuilding the previous version so the app stays usable..."
        if build_and_deploy; then
            log "✓ Previous version restored and rebuilt."
        else
            log "✗ Failed to rebuild the previous version. See ${LOG_FILE}."
        fi
    else
        log "No recorded source revision to restore."
    fi

    log "Upgrade rolled back. See ${LOG_FILE} for details."
    if [ -n "${DATA_BACKED_UP:-}" ] && [ -f "$DATA_BACKUP_FILE" ]; then
        log "Your pre-upgrade data backup is preserved at ${DATA_BACKUP_FILE}."
    fi
    exit 1
}

# Drop (or keep) the pre-upgrade data backup after a successful upgrade.
finalize_upgrade_backup() {
    [ -n "${DATA_BACKED_UP:-}" ] && [ -f "$DATA_BACKUP_FILE" ] || return 0
    if [ -n "${KEEP_BACKUP:-}" ]; then
        log "Pre-upgrade data backup kept at ${DATA_BACKUP_FILE}."
    else
        rm -f "$DATA_BACKUP_FILE"
        log "Removed pre-upgrade data backup (pass --keep-backup to retain it)."
    fi
}

run_install() {
    ensure_source_code

    # Initialize log file and build tools directory
    mkdir -p "$BUILD_TOOLS_DIR"
    : > "$LOG_FILE"

    build_and_deploy
    register_shell_cmd
    run_app
}

run_upgrade() {
    mkdir -p "$BUILD_TOOLS_DIR"
    : > "$LOG_FILE"

    echo ""
    echo "Existing Intel Geti installation detected at $WORK_DIR — running in UPGRADE mode."
    if ! confirm "Upgrade this installation to ${GIT_BRANCH}?"; then
        echo "Upgrade cancelled."
        exit 0
    fi

    # Snapshot the current state before anything changes.
    capture_rollback_point

    # From here on, any failure rolls the deployment back to the previous state.
    trap 'upgrade_rollback $LINENO' ERR

    ensure_source_code
    build_and_deploy

    if verify_app_start; then
        trap 'cleanup $LINENO' ERR
        finalize_upgrade_backup
        register_shell_cmd
        log "✓ Upgrade to ${GIT_BRANCH} completed successfully."
        run_app
    else
        # Verification failed → roll back (also reachable via the ERR trap).
        upgrade_rollback
    fi
}

main() {
    parse_args "$@"

    preflight_checks
    detect_upgrade

    if [ "$IS_UPGRADE" = true ]; then
        run_upgrade
    else
        run_install
    fi
}

run_app() {
    echo ""
    echo "Installation complete! Starting Intel Geti..."

    # Resolve the URL the user should open. The server binds to 0.0.0.0 by
    # default, which is not a valid address to open in a browser, so use
    # localhost. Honour PORT/HOST overrides if the user set them.
    local port="${PORT:-7860}"
    local browser_host="${HOST:-localhost}"
    if [ "$browser_host" = "0.0.0.0" ]; then
        browser_host="localhost"
    fi
    # The server terminates TLS itself (see app/main.py), so the scheme is https.
    local url="https://${browser_host}:${port}"

    echo ""
    echo "Geti will be available at: $url"
    echo "The server uses a self-signed certificate, so your browser will warn you"
    echo "about the connection the first time -- accept the warning to continue."
    echo ""

    cd "$WORK_DIR/application/backend"
    STATIC_FILES_DIR=html "$UV_DIR/uv" run app/main.py
}

main "$@"
