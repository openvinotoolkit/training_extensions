# Upgrading Intel Geti

This guide explains how to upgrade an existing Intel Geti installation to a newer
version while preserving your projects, datasets and models.

> [!IMPORTANT]
> An upgrade never deletes your data and prepares a backup before the migration.
> If migration fails, data can be easily restored.

## Table of contents

- [Upgrading Intel Geti](#upgrading-intel-geti)
  - [Table of contents](#table-of-contents)
  - [How upgrades work](#how-upgrades-work)
  - [Docker deployment](#docker-deployment)
  - [Windows desktop (MSIX) app](#windows-desktop-msix-app)
    - [Requirements for in-place upgrade to work](#requirements-for-in-place-upgrade-to-work)
  - [Source installation](#source-installation)
  - [What happens on failure](#what-happens-on-failure)
  - [Downgrading](#downgrading)
  - [Troubleshooting](#troubleshooting)

---

## How upgrades work

Geti stores all persistent state in a **data directory** (`DATA_DIR`, mounted as
the `geti-data` Docker volume or a per-user directory for the desktop app):

- a SQLite database `geti.db` (project/model/dataset metadata), and
- binary artifacts on the filesystem (media, model weights, dataset revisions).

Because migration runs on startup, **upgrading is simply a matter of replacing the
application with the newer version and starting it** — the data migration happens
by itself.

---

## Docker deployment

The Docker image bundles the whole application. The persistent `geti-data` and
`geti-logs` volumes are **not** part of the image, so replacing the image with a
newer one and reusing the same volumes preserves all your data. Because the
backend migrates data on startup, upgrading is a matter of pulling the newer image
and recreating the container against the same volumes.

> [!WARNING] take a snapshot of the `geti-data` volume before upgrading so
> you can restore the exact pre-upgrade state if needed.

```bash
# 1. (Recommended) Back up the data volume
docker run --rm -v geti-data:/data -v "$PWD":/backup alpine \
    tar czf /backup/geti-data-backup.tar.gz -C /data .

# 2. Pull the new image and retag it (available device choices: cpu, xpu, cuda)
docker pull ghcr.io/open-edge-platform/geti-cpu:3.2.0
docker tag  ghcr.io/open-edge-platform/geti-cpu:3.2.0 geti-cpu:latest

# 3. Recreate the container against the SAME volumes (data is migrated on startup)
just run-image --accelerator cpu --reload --detach

# 4. Verify (the backend serves /health over HTTPS with a self-signed cert)
curl -k https://localhost:7860/health
```

If the new container fails to become healthy or the backend exits with the
dedicated fatal-migration exit code `3`, retag the previous image, restore the
data snapshot, and recreate the container — see [Downgrading](#downgrading).

---

## Windows desktop (MSIX) app

The MSIX package contains the UI and the bundled backend. Your projects and models
live in a **per-user data directory** that is deliberately kept **outside** the
install location (`%LOCALAPPDATA%\Intel\Geti`), so it survives app updates.

To upgrade:

1. Download and run the newer `.msix` installer (or let Windows auto-update the
   package). Windows replaces the app in place, keeping your data directory.
2. Launch Geti. On first start the bundled backend migrates your data to the new
   version, taking a database backup first.

If the migration fails, the backend exits with the fatal code `3`.
The desktop app detects this and shows a **detailed error dialog** explaining
that the upgrade failed and where to find the logs, then closes. Because the
previous package can be reinstalled and backup is available, the app remains
usable — simply reinstall the previous `.msix` version (see
[Downgrading](#downgrading)) and restore the database from backup.

### Requirements for in-place upgrade to work

Windows performs an **in-place upgrade** (keeping the per-user data directory)
**only** when the newer package satisfies all of the following, otherwise it is
treated as a separate/side-by-side app or a same-version reinstall:

- **Identical `Identity/Name`** — must stay `intel.geti` across releases.
- **Identical `Publisher`** — must stay
  `CN=Intel Corporation, O=Intel Corporation, S=California, C=US` (this is bound
  to the signing certificate).
- **A strictly higher 4-part `Version`** in
  [`ui/src-tauri/msix/AppxManifest.xml`](../ui/src-tauri/msix/AppxManifest.xml)
  — e.g. `3.1.0.0` > `3.0.0.0`.

> **Maintainer note — bump the version for every release.** The
> [`AppxManifest.xml`](../ui/src-tauri/msix/AppxManifest.xml) `Version` must be
> incremented (4-part `Major.Minor.Build.Revision` form, e.g. `3.1.0.0`) for each
> release so the new package is recognised as an upgrade, while `Name` and
> `Publisher` are kept constant. The MSIX package itself is produced by the
> project's release/CI pipeline, not built locally.

---

## Source installation

For source installations there is a **single script** for both installing and
upgrading: [`install.sh`](../../install.sh) (Linux/macOS/WSL) and
[`install.ps1`](../../install.ps1) (Windows PowerShell). Re-running it on an
existing installation is automatically detected as an **upgrade** and performs a
safe, verifiable update with rollback:

```bash
# Linux / macOS / WSL — from the repository root
./install.sh
```

```powershell
# Windows PowerShell — from the repository root
.\install.ps1
```

When an existing installation is detected (a checkout with application data, or
`--upgrade` / `-Upgrade` forced), the script:

1. Records the current git revision and **backs up** the `application/backend/data`
   directory to `<work-dir>/.geti-upgrade-backups/` (skip with `--no-data-backup`
   / `-NoDataBackup` if you manage backups yourself).
2. Checks out the target release and rebuilds the backend and frontend.
3. **Verifies** the new version by starting it and waiting for
   `https://localhost:<port>/health`, detecting the backend's fatal-migration exit
   code `3`. The wait is bounded by `--health-timeout` / `-HealthTimeout` (default
   300s).
4. On success, launches the app (and drops the backup unless `--keep-backup` /
   `-KeepBackup` is given).
5. On **any** failure, restores the previous git revision,
   rebuilds the previous version, and leaves it usable.

All steps are written to `<work-dir>/.build/.install.log`. Useful options:

| Option (bash / PowerShell)                            | Purpose                                        |
| ----------------------------------------------------- | ---------------------------------------------- |
| `-u` / `--upgrade` &nbsp;·&nbsp; `-Upgrade`           | Force upgrade mode even without existing data  |
| `--no-data-backup` &nbsp;·&nbsp; `-NoDataBackup`      | Skip the pre-upgrade data backup               |
| `--keep-backup` &nbsp;·&nbsp; `-KeepBackup`           | Keep the backup after a successful upgrade     |
| `--backup-dir DIR` &nbsp;·&nbsp; `-BackupDir DIR`     | Where to store the backup                      |
| `--health-timeout N` &nbsp;·&nbsp; `-HealthTimeout N` | Seconds to wait for health before rolling back |
| `-y` / `--yes` &nbsp;·&nbsp; `-Yes`                   | Non-interactive (assume yes)                   |
| `-v` / `--verbose` &nbsp;·&nbsp; `-Verbose`           | Detailed output                                |

Run `./install.sh --help` (or `Get-Help .\install.ps1 -Detailed`) for the full
list.

---

## What happens on failure

A migration failure is treated as **fatal and non-restartable** — retrying would
fail identically — so the backend:

1. Logs detailed recovery guidance (including the backup location).
2. Exits with the dedicated code `3` so launchers/supervisors do **not** restart
   it in a loop.

## Downgrading

- **Docker:** point the container back at the previous image tag and, if needed,
  restore your data snapshot:

  ```bash
  docker rm -f geti-cpu
  docker run --rm -v geti-data:/data -v "$PWD":/backup alpine \
      sh -c 'rm -rf /data/* && tar xzf /backup/geti-data-backup.tar.gz -C /data'
  docker tag ghcr.io/open-edge-platform/geti-cpu:3.0.0 geti-cpu:latest
  just run-image --accelerator cpu --reload --detach true
  ```

- **MSIX:** uninstall the current package and install the previous `.msix`. Your
  per-user data directory is preserved. If you had upgraded and restored the database from backup,
  the data is already at the previous version.

---

## Troubleshooting

- **Where are the logs?**
  - Source install/upgrade: `<work-dir>/.build/.install.log`.
  - Docker: the container logs (`docker logs geti-cpu`) / the `geti-logs` volume.
  - Desktop: the per-user log directory (e.g. `%LOCALAPPDATA%\Intel\Geti`).
- **The new version won't start after an upgrade.** Check the logs for a
  `Fatal application upgrade error` message. Restore the database from the backup; start
  the previous version to continue, and open an issue at
  https://github.com/open-edge-platform/geti with the log attached.
- **A pre-upgrade database backup (`geti.db.<timestamp>.bak`) is left in the data
  directory.**
