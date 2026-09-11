# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Install OR upgrade the Intel Geti application and its dependencies on Windows.

.DESCRIPTION
    This script installs the Intel Geti application, including uv (Python package manager),
    Node.js/npm, and builds both the backend and frontend.

    Re-running the script on an existing installation upgrades it in place: the source is
    updated to the target version and the backend migrates your data on startup. Existing
    application data is backed up first and, if anything fails, the previous version and
    data are automatically restored so the app stays usable.

.PARAMETER Yes
    Assume yes to all prompts (non-interactive mode).

.PARAMETER WorkDir
    Set the working directory (default: .\geti).

.PARAMETER Upgrade
    Force upgrade mode even if no existing data is detected.

.PARAMETER NoDataBackup
    Skip the pre-upgrade data backup (NOT recommended; relies solely on the backend's
    own database rollback).

.PARAMETER KeepBackup
    Keep the pre-upgrade data backup after a successful upgrade.

.PARAMETER BackupDir
    Directory for pre-upgrade data backups (default: <WorkDir>\.geti-upgrade-backups).

.PARAMETER HealthTimeout
    Seconds to wait for the upgraded app to become healthy before rolling back (default: 300).

.PARAMETER GitBranch
    Override the git branch/tag to install (for testing purposes).

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Verbose -Yes
    .\install.ps1 -WorkDir "C:\my\custom\path"
    .\install.ps1 -Upgrade
#>

[CmdletBinding()]
param(
    [Alias("y")]
    [switch]$Yes,

    [Alias("w")]
    [string]$WorkDir = "$(Get-Location)\geti",

    [Alias("u")]
    [switch]$Upgrade,

    [switch]$NoDataBackup,

    [switch]$KeepBackup,

    [string]$BackupDir = "",

    [int]$HealthTimeout = 300,

    [string]$GitBranch = ""
)

$ErrorActionPreference = "Stop"

$GIT_URL = "https://github.com/open-edge-platform/geti.git"
# GIT_BRANCH can be overridden via the GIT_BRANCH environment variable or the
# -GitBranch parameter (for testing purposes).
$GIT_BRANCH = "app/v3.2.0"
if ($GitBranch) {
    $GIT_BRANCH = $GitBranch
}
elseif ($env:GIT_BRANCH) {
    $GIT_BRANCH = $env:GIT_BRANCH
}

# Exit code the backend uses for a fatal, non-restartable migration failure
# (see application/backend/app/lifecycle.py:MIGRATION_FATAL_EXIT_CODE). It lets
# the upgrade path distinguish "the data could not be migrated" from an ordinary
# crash so it can trigger a rollback.
$MIGRATION_FATAL_EXIT_CODE = 3

$BUILD_TOOLS_DIR = Join-Path $WorkDir ".build"
$UV_DIR = Join-Path $BUILD_TOOLS_DIR "uv"
$NVM_DIR = Join-Path $BUILD_TOOLS_DIR "nvm"
$LOG_FILE = Join-Path $BUILD_TOOLS_DIR ".install.log"

# Upgrade-related derived paths.
$DATA_PATH = Join-Path $WorkDir "application\backend\data"
if ([string]::IsNullOrEmpty($BackupDir)) {
    $BackupDir = Join-Path $WorkDir ".geti-upgrade-backups"
}
# Use [DateTime]::UtcNow rather than 'Get-Date -AsUTC' for Windows PowerShell 5.1 compatibility.
$script:Timestamp = ([DateTime]::UtcNow).ToString("yyyyMMddHHmmss")
$script:DataBackupPath = Join-Path $BackupDir "geti-data-$($script:Timestamp)"
# Populated during an upgrade so the rollback path knows what to restore.
$script:PreviousSha = ""
$script:DataBackedUp = $false

$script:NPM_BIN = ""

function Write-Step {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

# Echo a timestamped message to the console and append it to the log file. Used
# by the upgrade path so the sequence of upgrade/rollback actions is captured
# for troubleshooting.
function Write-Log {
    param([string]$Message)
    $line = "{0} {1}" -f ([DateTime]::UtcNow).ToString("HH:mm:ss"), $Message
    Write-Host $line
    if ($LOG_FILE -and (Test-Path (Split-Path $LOG_FILE -Parent))) {
        Add-Content -Path $LOG_FILE -Value $line -ErrorAction SilentlyContinue
    }
}

function Confirm-Prompt {
    param([string]$Prompt)

    if ($Yes) { return $true }

    $response = Read-Host "$Prompt [Y/n]"
    if ($response -match "^n(o)?$") { return $false }
    return $true
}

function Invoke-Cmd {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    # Temporarily allow stderr output without terminating (tools like npm/git
    # write warnings to stderr even on success).
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        if ($VerbosePreference -eq "Continue") {
            & $Command @Arguments 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    Write-Host $_.ToString() -ForegroundColor Yellow
                } else {
                    Write-Host $_
                }
            }
        } else {
            & $Command @Arguments *>> $LOG_FILE
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }

    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Command '$Command $($Arguments -join ' ')' failed with exit code $LASTEXITCODE"
    }
}

function Invoke-CmdSpinner {
    # Run a long command quietly (output to the log file) while showing an
    # animated spinner, so the step never looks frozen. In verbose mode the
    # full output is streamed instead.
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string]$Activity = "Working"
    )

    if ($VerbosePreference -eq "Continue") {
        Write-Host "$Activity..."
        Invoke-Cmd -Command $Command -Arguments $Arguments
        return
    }

    $stdoutTmp = [System.IO.Path]::GetTempFileName()
    $stderrTmp = [System.IO.Path]::GetTempFileName()

    try {
        $proc = Start-Process -FilePath $Command -ArgumentList $Arguments `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdoutTmp -RedirectStandardError $stderrTmp

        # Touching .Handle caches the process handle so .ExitCode is reliably
        # populated after exit. Without this, Start-Process returns $null for
        # .ExitCode when launching .cmd/.bat files (e.g. npm.cmd), which would
        # be misread as a failure.
        $null = $proc.Handle

        $spinner = '|', '/', '-', '\'
        $i = 0
        while (-not $proc.HasExited) {
            Write-Host -NoNewline ("`r{0}... {1}" -f $Activity, $spinner[$i % 4])
            Start-Sleep -Milliseconds 200
            $i++
        }
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode

        # Append captured output to the log file for troubleshooting.
        Get-Content -LiteralPath $stdoutTmp -ErrorAction SilentlyContinue | Add-Content -LiteralPath $LOG_FILE
        Get-Content -LiteralPath $stderrTmp -ErrorAction SilentlyContinue | Add-Content -LiteralPath $LOG_FILE

        if ($exitCode -ne 0) {
            Write-Host ("`r{0}... failed " -f $Activity) -ForegroundColor Red
            throw "Command '$Command $($Arguments -join ' ')' failed with exit code $exitCode"
        }

        Write-Host ("`r{0}... done   " -f $Activity) -ForegroundColor Green
    } finally {
        Remove-Item -LiteralPath $stdoutTmp, $stderrTmp -ErrorAction SilentlyContinue
    }
}

function Get-RequiredUvVersion {
    $pyprojectPath = Join-Path $WorkDir "application\backend\pyproject.toml"
    $content = Get-Content $pyprojectPath -Raw

    if ($content -match '\[tool\.uv\][\s\S]*?required-version\s*=\s*"[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }

    throw "Could not parse uv version from pyproject.toml"
}

function Get-RequiredNodeVersion {
    $packageJsonPath = Join-Path $WorkDir "application\ui\package.json"
    $json = Get-Content $packageJsonPath -Raw | ConvertFrom-Json

    $nodeConstraint = $json.engines.node
    if ($nodeConstraint -match '>=v?([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }

    throw "Could not parse node version from package.json"
}

function Get-RequiredNpmVersion {
    $packageJsonPath = Join-Path $WorkDir "application\ui\package.json"
    $json = Get-Content $packageJsonPath -Raw | ConvertFrom-Json

    $npmConstraint = $json.engines.npm
    if ($npmConstraint -match '>=([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }

    throw "Could not parse npm version from package.json"
}

function Install-Uv {
    $uvVersion = Get-RequiredUvVersion
    $uvExe = Join-Path $UV_DIR "uv.exe"

    if (Test-Path $uvExe) {
        $installedVersion = & $uvExe --version | ForEach-Object { ($_ -split ' ')[1] }
        if ($installedVersion -eq $uvVersion) {
            Write-Step "uv $uvVersion found in $UV_DIR"
            return
        } else {
            Write-Step "uv version mismatch: installed=$installedVersion, required=$uvVersion. Reinstalling..."
        }
    }

    Write-Step "Installing uv $uvVersion to: $UV_DIR"
    if (-not (Confirm-Prompt "Would you like to install uv now?")) {
        throw "uv installation skipped. Cannot continue without uv."
    }

    if (-not (Test-Path $UV_DIR)) {
        New-Item -ItemType Directory -Path $UV_DIR -Force | Out-Null
    }

    $installerUrl = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-installer.ps1"
    $env:UV_INSTALL_DIR = $UV_DIR

    Invoke-CmdSpinner -Command "powershell" `
        -Arguments @("-ExecutionPolicy", "Bypass", "-Command", "irm '$installerUrl' | iex") `
        -Activity "Downloading and installing uv $uvVersion"

    Remove-Item Env:\UV_INSTALL_DIR -ErrorAction SilentlyContinue

    if (-not (Test-Path $uvExe)) {
        throw "uv installation failed. Expected binary at $uvExe."
    }

    Write-Step "uv installation complete."
}

function Install-Nvm {
    $nvmExe = Join-Path $NVM_DIR "nvm.exe"

    if (Test-Path $nvmExe) {
        Write-Step "nvm found in $NVM_DIR."
        return
    }

    Write-Step "Installing nvm-windows to: $NVM_DIR"
    if (-not (Confirm-Prompt "Would you like to install nvm-windows now?")) {
        throw "nvm installation skipped. Cannot continue without nvm."
    }

    if (-not (Test-Path $NVM_DIR)) {
        New-Item -ItemType Directory -Path $NVM_DIR -Force | Out-Null
    }

    # Download nvm-windows noinstall zip
    $nvmVersion = "1.2.2"
    $nvmZipUrl = "https://github.com/coreybutler/nvm-windows/releases/download/$nvmVersion/nvm-noinstall.zip"
    $nvmZipPath = Join-Path $BUILD_TOOLS_DIR "nvm-noinstall.zip"

    Write-Host "Downloading nvm-windows $nvmVersion..."
    $iwrParams = @{ Uri = $nvmZipUrl; OutFile = $nvmZipPath }
    # -UseBasicParsing is required in Windows PowerShell 5.1 to avoid IE engine dependency.
    # In PowerShell 7+ basic parsing is the default and the parameter is accepted but ignored.
    if ($PSVersionTable.PSVersion.Major -le 5) {
        $iwrParams["UseBasicParsing"] = $true
    }
    Invoke-WebRequest @iwrParams

    Expand-Archive -Path $nvmZipPath -DestinationPath $NVM_DIR -Force
    Remove-Item $nvmZipPath -Force

    # Configure nvm settings
    $nodeDir = Join-Path $NVM_DIR "nodejs"
    $settingsContent = @"
root: $NVM_DIR
path: $nodeDir
"@
    Set-Content -Path (Join-Path $NVM_DIR "settings.txt") -Value $settingsContent -Encoding ASCII

    Write-Step "nvm-windows installation complete."
}

function Install-Npm {
    $requiredNodeVersion = Get-RequiredNodeVersion
    $requiredNpmVersion = Get-RequiredNpmVersion
    $nvmExe = Join-Path $NVM_DIR "nvm.exe"
    $nodeDir = Join-Path $NVM_DIR "nodejs"
    $nodeVersionDir = Join-Path $NVM_DIR "v$requiredNodeVersion"

    # Check if the required node version is already installed
    $nodeBin = Join-Path $nodeVersionDir "node.exe"
    $npmBin = Join-Path $nodeVersionDir "npm.cmd"

    if (Test-Path $nodeBin) {
        if (-not (Test-Path $npmBin)) {
            throw "node.exe found at $nodeBin but npm.cmd is missing at $npmBin. Remove $nodeVersionDir and re-run the installer."
        }
        $script:NPM_BIN = $npmBin
        $env:PATH = "$nodeVersionDir;$env:PATH"
        $installedNpmVersion = & $npmBin --version 2>$null
        if ($installedNpmVersion -and ([version]$installedNpmVersion -ge [version]$requiredNpmVersion)) {
            Write-Step "node $requiredNodeVersion and npm $installedNpmVersion found."
            return
        }

        Write-Step "npm version too old: installed=$installedNpmVersion, required>=$requiredNpmVersion. Upgrading..."
        Invoke-Cmd -Command $npmBin -Arguments @("install", "-g", "npm@$requiredNpmVersion")
        return
    }

    Write-Step "Required node $requiredNodeVersion not found. Installing..."

    if (-not (Confirm-Prompt "Would you like to install node/npm now?")) {
        throw "node/npm installation skipped. Cannot continue without node/npm."
    }

    # Set NVM_HOME for nvm.exe to work properly
    $env:NVM_HOME = $NVM_DIR
    $env:NVM_SYMLINK = $nodeDir

    # Install node (nvm install does not require elevation)
    Invoke-CmdSpinner -Command $nvmExe -Arguments @("install", $requiredNodeVersion) `
        -Activity "Downloading and installing node $requiredNodeVersion"

    # Skip "nvm use" as it requires admin elevation to create a symlink.
    # Instead, we reference binaries directly from the version-specific directory
    # and prepend to PATH so node/npm can find each other.
    $env:PATH = "$nodeVersionDir;$env:PATH"

    $script:NPM_BIN = $npmBin

    if (-not (Test-Path $npmBin)) {
        throw "node installation succeeded but npm.cmd not found at $npmBin. Installation may be corrupt."
    }

    $installedNpmVersion = & $npmBin --version 2>$null
    if ($installedNpmVersion -and ([version]$installedNpmVersion -lt [version]$requiredNpmVersion)) {
        Invoke-Cmd -Command $npmBin -Arguments @("install", "-g", "npm@$requiredNpmVersion")
    }

    Write-Step "node/npm installation complete."
}

function Find-NvidiaGpus {
    $gpuCount = 0

    # Try nvidia-smi
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            $gpus = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($gpus) {
                $gpuCount = ($gpus | Measure-Object -Line).Lines
                if ($gpuCount -gt 0) {
                    Write-Step "Detected $gpuCount NVIDIA GPU(s) via nvidia-smi:"
                    & nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
                    return $true
                }
            }
        } catch {}
    }

    # Try WMI/CIM
    try {
        $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
        if ($gpus) {
            $gpuCount = @($gpus).Count
            Write-Step "Detected $gpuCount NVIDIA GPU(s):"
            $gpus | ForEach-Object { Write-Host "  $($_.Name)" }
            return $true
        }
    } catch {}

    Write-Host "No NVIDIA GPUs detected."
    return $false
}

function Find-IntelGpus {
    # Try WMI/CIM
    try {
        $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "Intel" -and $_.Name -match "Arc" }
        if ($gpus) {
            $gpuCount = @($gpus).Count
            Write-Step "Detected $gpuCount Intel GPU(s):"
            $gpus | ForEach-Object { Write-Host "  $($_.Name)" }
            return $true
        }
    } catch {}

    Write-Host "No Intel GPUs detected."
    return $false
}

function Test-PreflightChecks {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is not installed. Please install git and try again."
    }
}

function Invoke-EnsureSourceCode {
    # Git commands write informational messages to stderr which PowerShell
    # treats as terminating errors under $ErrorActionPreference = "Stop".
    # We temporarily switch to Continue for all git invocations here.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        if (-not (Test-Path $WorkDir)) {
            Write-Step "Cloning Intel Geti repository from $GIT_URL..."
            Write-Host "This can take several minutes depending on your connection." -ForegroundColor DarkGray
            # Let git print its native progress meter so the clone never looks frozen.
            & git -c advice.detachedHead=false clone --progress --branch $GIT_BRANCH $GIT_URL $WorkDir
            if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit code $LASTEXITCODE)" }
        } else {
            Write-Step "Work directory $WorkDir already exists, skipping clone."

            $remoteUrl = (& git -C $WorkDir remote get-url origin 2>$null)
            if ($remoteUrl -ne $GIT_URL) {
                throw "$WorkDir remote origin is '$remoteUrl', expected '$GIT_URL'. Remove $WorkDir and re-run the installer."
            }

            $currentSha = (& git -C $WorkDir rev-parse HEAD 2>$null)
            & git -C $WorkDir fetch origin "refs/tags/${GIT_BRANCH}:refs/tags/${GIT_BRANCH}" --force 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                # Fallback: try fetching as a branch
                & git -C $WorkDir fetch origin $GIT_BRANCH --tags 2>&1 | Out-Null
            }
            # Resolve the expected SHA: try as tag first, then as remote branch
            $expectedSha = (& git -C $WorkDir rev-parse "refs/tags/$GIT_BRANCH" 2>$null) | Select-Object -First 1
            if (-not $expectedSha -or $expectedSha -notmatch '^[0-9a-f]{40}$') {
                $expectedSha = (& git -C $WorkDir rev-parse "origin/$GIT_BRANCH" 2>$null) | Select-Object -First 1
            }
            if (-not $expectedSha -or $expectedSha -notmatch '^[0-9a-f]{40}$') {
                throw "Could not resolve ref '$GIT_BRANCH'. Ensure it exists on the remote."
            }

            if ($currentSha -ne $expectedSha) {
                Write-Step "Updating to $GIT_BRANCH..."
                & git -c advice.detachedHead=false -C $WorkDir checkout --force $GIT_BRANCH 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "git checkout failed (exit code $LASTEXITCODE)" }
            }
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}

function Install-BuildTools {
    Install-Uv
    Install-Nvm
    Install-Npm
}

function Find-Hardware {
    $script:HAS_NVIDIA_GPU = $false
    $script:HAS_INTEL_GPU = $false

    if (Find-NvidiaGpus) {
        $script:HAS_NVIDIA_GPU = $true
    }

    if (Find-IntelGpus) {
        $script:HAS_INTEL_GPU = $true
    }

    if ($script:HAS_NVIDIA_GPU) {
        $script:ACCELERATOR = "cuda"
    } elseif ($script:HAS_INTEL_GPU) {
        $script:ACCELERATOR = "xpu"
    } else {
        $script:ACCELERATOR = "cpu"
    }

    $env:ACCELERATOR = $script:ACCELERATOR
}

function Build-Backend {
    Write-Step "Building Python environment using accelerator: $($script:ACCELERATOR)"
    Write-Host "This downloads PyTorch, OpenVINO and other large packages and can take several minutes." -ForegroundColor DarkGray
    $backendDir = Join-Path $WorkDir "application\backend"
    Push-Location $backendDir

    try {
        $uvExe = Join-Path $UV_DIR "uv.exe"

        # uv shows its own progress meter; do not suppress it so the user gets feedback.
        & $uvExe sync --frozen --extra mqtt --extra $script:ACCELERATOR

        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

        Write-Step "Generating OpenAPI specification..."
        $prevPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = "."
        try {
            & $uvExe run --no-sync app/cli.py gen-api --target-path openapi.json
            if ($LASTEXITCODE -ne 0) { throw "OpenAPI generation failed" }
        } finally {
            $env:PYTHONPATH = $prevPythonPath
        }

        $uiApiDir = Join-Path $WorkDir "application\ui\src\api"
        Copy-Item -Path "openapi.json" -Destination (Join-Path $uiApiDir "openapi-spec.json") -Force
    } finally {
        Pop-Location
    }
}

function Build-Frontend {
    $uiDir = Join-Path $WorkDir "application\ui"
    Push-Location $uiDir

    try {
        $env:npm_config_yes = "true"

        # Remove build artifacts and cloned workspace packages left over from a
        # previous build/version. `git checkout --force` does not touch these
        # untracked paths, and stale contents make `npm ci` fail with
        # "package.json and package-lock.json are not in sync" (e.g. a missing
        # @geti/ui workspace package from the old revision).
        foreach ($stale in @("node_modules", "packages", "dist")) {
            $stalePath = Join-Path $uiDir $stale
            if (Test-Path $stalePath) {
                Remove-Item -Path $stalePath -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        Invoke-CmdSpinner -Command $script:NPM_BIN `
            -Arguments @("ci", "--foreground-scripts") `
            -Activity "Installing UI dependencies (this may take several minutes)"

        Invoke-CmdSpinner -Command $script:NPM_BIN `
            -Arguments @("run", "build:api") `
            -Activity "Building API client"

        $env:ASSET_PREFIX = "/html"
        Invoke-CmdSpinner -Command $script:NPM_BIN `
            -Arguments @("run", "build") `
            -Activity "Building UI (this may take several minutes)"
        Remove-Item Env:\ASSET_PREFIX -ErrorAction SilentlyContinue
    } finally {
        Pop-Location
    }
}

function Deploy-Frontend {
    $htmlDir = Join-Path $WorkDir "application\backend\html"

    Write-Step "Copying built UI to backend html directory..."
    if (Test-Path $htmlDir) {
        Remove-Item -Path $htmlDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $htmlDir -Force | Out-Null

    $distDir = Join-Path $WorkDir "application\ui\dist\*"
    Copy-Item -Path $distDir -Destination $htmlDir -Recurse -Force
}

function Register-ShellCommand {
    $uvExe = Join-Path $UV_DIR "uv.exe"
    $backendDir = Join-Path $WorkDir "application\backend"

    # Create a geti.cmd batch file in the work directory
    $cmdPath = Join-Path $WorkDir "geti.cmd"
    $cmdContent = @"
@echo off
pushd "$backendDir"
set STATIC_FILES_DIR=html
"$uvExe" run app/main.py %*
popd
"@
    Set-Content -Path $cmdPath -Value $cmdContent -Encoding ASCII

    # Create a geti.ps1 PowerShell wrapper
    $ps1Path = Join-Path $WorkDir "geti.ps1"
    $ps1Content = @"
# Intel Geti launcher
param([Parameter(ValueFromRemainingArguments=`$true)]`$Args)
Push-Location "$backendDir"
try {
    `$env:STATIC_FILES_DIR = "html"
    & "$uvExe" run app/main.py @Args
} finally {
    Pop-Location
}
"@
    Set-Content -Path $ps1Path -Value $ps1Content

    # Add to PowerShell profile (opt-in: requires confirmation or -Yes)
    if (-not (Confirm-Prompt "Would you like to add the 'geti' function to your PowerShell profile?")) {
        Write-Host "Profile modification skipped."
        Write-Host "You can run geti manually via: $ps1Path"
        Write-Host "Or via batch file: $cmdPath"
        return
    }

    $profileDir = Split-Path $PROFILE -Parent
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Path $PROFILE -Force | Out-Null
    }

    $beginMarker = "# BEGIN Intel Geti"
    $endMarker = "# END Intel Geti"
    $profileContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue

    # Remove old marker block if present
    if ($profileContent -and $profileContent -match [regex]::Escape($beginMarker)) {
        $profileContent = $profileContent -replace "(?s)\r?\n?$([regex]::Escape($beginMarker)).*?$([regex]::Escape($endMarker))\r?\n?", ""
        Set-Content -Path $PROFILE -Value $profileContent -NoNewline -Encoding UTF8
    }

    $functionBlock = @"

$beginMarker
function geti { Push-Location "$backendDir"; try { `$env:STATIC_FILES_DIR = "html"; & "$uvExe" run app/main.py @args } finally { Pop-Location } }
$endMarker
"@
    Add-Content -Path $PROFILE -Value $functionBlock -Encoding UTF8

    Write-Step "Function 'geti' written to $PROFILE"
    Write-Host "Run '. `$PROFILE' to activate it in the current session."
    Write-Host "Example: `$env:HOST='0.0.0.0'; `$env:PORT='8080'; geti"
    Write-Host ""
    Write-Host "Batch file also available at: $cmdPath"
}

# ─── Upgrade support ─────────────────────────────────────────────────────────

function Invoke-BuildAndDeploy {
    Install-BuildTools
    Find-Hardware
    Build-Backend
    Build-Frontend
    Deploy-Frontend
}

# Decide whether this run is a fresh install or an in-place upgrade. An upgrade
# is any run against an existing source checkout that already holds application
# data (or when the user forces it with -Upgrade). A checkout with no data is
# still treated as a fresh install so first-time builds are never gated behind
# the (heavier) upgrade path.
function Test-UpgradeMode {
    if (-not (Test-Path (Join-Path $WorkDir ".git"))) { return $false }
    if ($Upgrade) { return $true }
    if ((Test-Path $DATA_PATH) -and (Get-ChildItem -Path $DATA_PATH -Force -ErrorAction SilentlyContinue)) {
        return $true
    }
    return $false
}

# Record everything needed to restore the current version: the git revision and
# a full snapshot of the application data directory.
function Save-RollbackPoint {
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $script:PreviousSha = (& git -C $WorkDir rev-parse HEAD 2>$null | Select-Object -First 1)
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($script:PreviousSha) {
        Write-Log "Recorded current source version: $($script:PreviousSha)"
    } else {
        Write-Log "WARNING: could not determine current git revision; source rollback disabled."
    }

    if ($NoDataBackup) {
        Write-Log "WARNING: -NoDataBackup set; skipping pre-upgrade data backup."
        return
    }
    if (-not (Test-Path $DATA_PATH) -or -not (Get-ChildItem -Path $DATA_PATH -Force -ErrorAction SilentlyContinue)) {
        Write-Log "No existing application data found; nothing to back up."
        return
    }

    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    Write-Log "Backing up application data -> $($script:DataBackupPath) ..."
    Write-Log "(This may take a while and disk space proportional to your data size.)"
    Copy-Item -Path $DATA_PATH -Destination $script:DataBackupPath -Recurse -Force
    $script:DataBackedUp = $true
    Write-Log "OK Data backup created."
}

# The backend serves /health over HTTPS with a self-signed cert, so certificate
# validation must be bypassed. Prefer curl.exe (ships with Windows 10 1803+);
# fall back to Invoke-WebRequest with a cert-check bypass.
function Test-AppHealth {
    param([string]$Port)
    $url = "https://localhost:${Port}/health"

    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        # curl.exe returns a non-zero exit code (e.g. 7 "could not connect")
        # while the backend is still starting up. Under $ErrorActionPreference =
        # 'Stop' combined with $PSNativeCommandUseErrorActionPreference (on by
        # default in PowerShell 7.3+), that non-zero exit is turned into a
        # terminating error that would abort the whole health-polling loop on the
        # very first probe -- and be misreported as an "Upgrade error" that
        # triggers a rollback. Relax the preference locally so a failed probe just
        # means "not healthy yet" and the loop keeps retrying until the timeout.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & curl.exe -ksSf --max-time 5 $url *> $null
            return ($LASTEXITCODE -eq 0)
        } finally {
            $ErrorActionPreference = $prevEAP
        }
    }

    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $null = Invoke-WebRequest -Uri $url -TimeoutSec 5 -SkipCertificateCheck -UseBasicParsing
        } else {
            $prevCallback = [System.Net.ServicePointManager]::ServerCertificateValidationCallback
            [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
            try {
                $null = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing
            } finally {
                [System.Net.ServicePointManager]::ServerCertificateValidationCallback = $prevCallback
            }
        }
        return $true
    } catch {
        return $false
    }
}

# Start the freshly built backend and wait until it reports healthy, so a failed
# data migration is caught before the upgrade is considered successful. The
# verification instance is stopped once healthy; the app is (re)started normally
# by Start-App afterwards.
function Invoke-VerifyAppStart {
    $port = if ($env:PORT) { $env:PORT } else { "7860" }
    Write-Log "Verifying the upgraded application starts and migrates data (up to ${HealthTimeout}s)..."

    $uvExe = Join-Path $UV_DIR "uv.exe"
    $backendDir = Join-Path $WorkDir "application\backend"
    $stdoutTmp = [System.IO.Path]::GetTempFileName()
    $stderrTmp = [System.IO.Path]::GetTempFileName()

    # uv spawns a child python process; kill the whole tree so nothing keeps
    # holding the port before Start-App rebinds it.
    $stopTree = {
        param($p)
        if (-not $p) { return }
        try { & taskkill.exe /PID $p.Id /T /F *> $null } catch {}
        try { if (-not $p.HasExited) { $p.Kill() } } catch {}
        try { $p.WaitForExit(10000) | Out-Null } catch {}
    }

    $env:STATIC_FILES_DIR = "html"
    try {
        $proc = Start-Process -FilePath $uvExe -ArgumentList @("run", "app/main.py") `
            -WorkingDirectory $backendDir -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdoutTmp -RedirectStandardError $stderrTmp
        # Cache the handle so .ExitCode is reliably populated after exit.
        $null = $proc.Handle

        $deadline = (Get-Date).AddSeconds($HealthTimeout)
        while ((Get-Date) -lt $deadline) {
            if ($proc.HasExited) {
                $code = $proc.ExitCode
                if ($code -eq $MIGRATION_FATAL_EXIT_CODE) {
                    Write-Log "FAIL Backend exited with fatal migration code ${code}: data could not be migrated."
                } else {
                    Write-Log "FAIL Backend exited unexpectedly (exit code ${code}) during verification."
                }
                return $false
            }
            if (Test-AppHealth -Port $port) {
                Write-Log "OK Upgraded version is healthy."
                & $stopTree $proc
                return $true
            }
            Start-Sleep -Seconds 3
        }

        Write-Log "FAIL Timed out waiting for the upgraded app to become healthy."
        & $stopTree $proc
        return $false
    } finally {
        Get-Content -LiteralPath $stdoutTmp -ErrorAction SilentlyContinue | Add-Content -LiteralPath $LOG_FILE
        Get-Content -LiteralPath $stderrTmp -ErrorAction SilentlyContinue | Add-Content -LiteralPath $LOG_FILE
        Remove-Item -LiteralPath $stdoutTmp, $stderrTmp -ErrorAction SilentlyContinue
    }
}

# Restore the previous version (source + data) and rebuild it so the app remains
# usable after a failed upgrade.
function Invoke-UpgradeRollback {
    $ErrorActionPreference = "Continue"
    Write-Host ""
    Write-Log "----------------------------------------------"
    Write-Log "Upgrade failed. Rolling back to the previous version..."

    if ($script:DataBackedUp -and (Test-Path $script:DataBackupPath)) {
        Write-Log "Restoring application data from backup..."
        try {
            if (Test-Path $DATA_PATH) { Remove-Item -Path $DATA_PATH -Recurse -Force }
            Copy-Item -Path $script:DataBackupPath -Destination $DATA_PATH -Recurse -Force
            Write-Log "OK Data restored to its pre-upgrade state."
        } catch {
            Write-Log "FAIL Could not restore data. Your backup is preserved at $($script:DataBackupPath)."
        }
    }

    if ($script:PreviousSha) {
        Write-Log "Restoring previous source version ($($script:PreviousSha))..."
        & git -c advice.detachedHead=false -C $WorkDir checkout --force $script:PreviousSha 2>&1 | Out-Null
        Write-Log "Rebuilding the previous version so the app stays usable..."
        try {
            Invoke-BuildAndDeploy
            Write-Log "OK Previous version restored and rebuilt."
        } catch {
            Write-Log "FAIL Failed to rebuild the previous version. See $LOG_FILE."
        }
    } else {
        Write-Log "No recorded source revision to restore."
    }

    Write-Log "Upgrade rolled back. See $LOG_FILE for details."
    if ($script:DataBackedUp -and (Test-Path $script:DataBackupPath)) {
        Write-Log "Your pre-upgrade data backup is preserved at $($script:DataBackupPath)."
    }
}

# Drop (or keep) the pre-upgrade data backup after a successful upgrade.
function Complete-UpgradeBackup {
    if (-not ($script:DataBackedUp -and (Test-Path $script:DataBackupPath))) { return }
    if ($KeepBackup) {
        Write-Log "Pre-upgrade data backup kept at $($script:DataBackupPath)."
    } else {
        Remove-Item -Path $script:DataBackupPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Log "Removed pre-upgrade data backup (pass -KeepBackup to retain it)."
    }
}

function Initialize-Logging {
    if (-not (Test-Path $BUILD_TOOLS_DIR)) {
        New-Item -ItemType Directory -Path $BUILD_TOOLS_DIR -Force | Out-Null
    }
    "" | Set-Content -Path $LOG_FILE
}

function Invoke-Install {
    Invoke-EnsureSourceCode
    Initialize-Logging

    Invoke-BuildAndDeploy
    Register-ShellCommand
    Start-App
}

function Invoke-Upgrade {
    # The checkout already exists, so logging can be initialized up front.
    Initialize-Logging

    Write-Host ""
    Write-Host "Existing Intel Geti installation detected at $WorkDir - running in UPGRADE mode." -ForegroundColor Cyan
    if (-not (Confirm-Prompt "Upgrade this installation to $GIT_BRANCH?")) {
        Write-Host "Upgrade cancelled."
        return
    }

    # Snapshot the current state before anything changes.
    Save-RollbackPoint

    try {
        Invoke-EnsureSourceCode
        Invoke-BuildAndDeploy

        if (Invoke-VerifyAppStart) {
            Complete-UpgradeBackup
            Register-ShellCommand
            Write-Log "OK Upgrade to $GIT_BRANCH completed successfully."
            Start-App
        } else {
            Invoke-UpgradeRollback
            exit 1
        }
    } catch {
        Write-Log "FAIL Upgrade error: $_"
        Invoke-UpgradeRollback
        exit 1
    }
}

# ─── Main ────────────────────────────────────────────────────────────────────

function Main {
    Write-Host ""
    Write-Host "Intel Geti Installer (Windows/PowerShell)" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""

    Test-PreflightChecks

    if (Test-UpgradeMode) {
        Invoke-Upgrade
    } else {
        Invoke-Install
    }
}

function Start-App {
    Write-Host ""
    Write-Step "Installation complete! Starting Intel Geti..."

    $uvExe = Join-Path $UV_DIR "uv.exe"
    $backendDir = Join-Path $WorkDir "application\backend"

    # Resolve the URL the user should open. The server binds to 0.0.0.0 by
    # default, which is not a valid address to open in a browser, so use
    # localhost. Honour PORT/HOST overrides if the user set them.
    $port = if ($env:PORT) { $env:PORT } else { "7860" }
    $browserHost = if ($env:HOST -and $env:HOST -ne "0.0.0.0") { $env:HOST } else { "localhost" }
    # The server terminates TLS itself (see app/main.py), so the scheme is https.
    $url = "https://${browserHost}:${port}"

    Write-Host ""
    Write-Host "Geti will be available at: " -NoNewline
    Write-Host $url -ForegroundColor Cyan
    Write-Host "The server uses a self-signed certificate, so your browser will warn you"
    Write-Host "about the connection the first time -- accept the warning to continue."
    Write-Host ""

    Push-Location $backendDir
    try {
        $env:STATIC_FILES_DIR = "html"
        & $uvExe run app/main.py
    } finally {
        Pop-Location
    }
}

try {
    Main
} catch {
    Write-Host ""
    Write-ErrorMessage "Installation failed: $_"
    if (Test-Path $LOG_FILE) {
        Write-Host "Check $LOG_FILE for details."
    }
    Write-Host "Re-run with -Verbose for more details."
    exit 1
}

