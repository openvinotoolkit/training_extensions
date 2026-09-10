// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//! Side-car backend lifecycle: locating, configuring and spawning the
//! PyInstaller-frozen `geti-backend` next to the Tauri executable.

use std::{
    env,
    path::PathBuf,
    process::{Child, Command},
};

use tauri::{AppHandle, Manager};

/// A spawned side-car together with the OS-level containment that guarantees it
/// (and its multiprocessing workers) cannot outlive the shell.
pub struct Sidecar {
    /// The child process handle. Owning it — instead of only its PID — is what
    /// keeps the PID reserved: a reaped PID can be recycled by Windows, and
    /// killing "the backend" by a stale PID could take down an unrelated
    /// process tree.
    pub child: Child,
    /// Windows job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Dropping it
    /// (or crashing) terminates the whole backend tree; see [`crate::job`].
    #[cfg(windows)]
    pub job: Option<crate::job::JobHandle>,
}

/// "geti-backend.exe" on Windows, "geti-backend" elsewhere.
fn backend_filename() -> &'static str {
    if cfg!(windows) {
        "geti-backend.exe"
    } else {
        "geti-backend"
    }
}

/// Spawns the side-car backend.
///
/// Layout depends on packaging:
///
/// - **`tauri dev`** stages `geti-backend` + `_internal/` flat next to the
///   Tauri executable in `target/<profile>/`. PyInstaller's bootloader sees
///   it's not inside a `.app` and uses the sibling `_internal/` directly.
/// - **`tauri build` (.app on macOS)** moves them into a `backend/`
///   subdirectory under `Contents/MacOS/`. This is required because if the
///   PyInstaller-frozen executable lives directly at
///   `<Bundle>.app/Contents/MacOS/<exe>` the bootloader switches to "macOS
///   bundle" mode and looks for shared libraries under `Contents/Frameworks/`
///   — which Tauri doesn't populate, so launch fails with
///   `Failed to load Python shared library 'libpython*.dylib'`. Putting the
///   sidecar one directory deeper avoids the bundle detection and keeps the
///   standalone `_internal/` layout working. The post-build move is done by
///   the packaging recipe in [`application/Justfile`](../../Justfile).
/// - **Other platforms** keep the flat layout in both dev and release because
///   no equivalent bundle-detection exists on Windows / Linux.
pub fn spawn_backend(app: &AppHandle) -> std::io::Result<Sidecar> {
    let exe_path = env::current_exe().expect("failed to get current exe path");
    let exe_dir = exe_path
        .parent()
        .expect("failed to get parent directory of exe");
    let backend_path = locate_backend(exe_dir);

    log::info!("▶ Looking for backend side-car at {:?}", backend_path);
    let mut command = Command::new(&backend_path);
    apply_default_env(&mut command, app);

    // Release builds suppress the side-car's console window; in dev it stays
    // visible so the backend's logs show up in the terminal.
    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    // Put the backend into its own process group so that on shutdown we can
    // `kill -- -<pid>` to terminate it together with all worker processes it
    // spawns via Python multiprocessing.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }

    let child = command.spawn()?;

    // Belt-and-suspenders containment: even if the shell is killed outright
    // (Task Manager, power loss of the UI process, panic) the kernel tears the
    // backend tree down with us, so no orphaned worker keeps the port bound or
    // the SQLite database locked.
    #[cfg(windows)]
    let job = crate::job::contain(&child);

    log::info!("▶ Spawned backend: {:?}", backend_path);
    Ok(Sidecar {
        child,
        #[cfg(windows)]
        job,
    })
}

/// Resolve the side-car path. Prefers `<exe_dir>/backend/<name>` (release
/// `.app` layout) and falls back to `<exe_dir>/<name>` (dev / non-macOS).
fn locate_backend(exe_dir: &std::path::Path) -> PathBuf {
    let nested = exe_dir.join("backend").join(backend_filename());
    if nested.exists() {
        return nested;
    }
    exe_dir.join(backend_filename())
}

/// Apply the default environment for the side-car. Each var is only set if the
/// inherited environment doesn't already provide one, so callers can override.
fn apply_default_env(command: &mut Command, app: &AppHandle) {
    // The Tauri 2 webview loads the UI from `tauri://localhost` on macOS/Linux
    // and `https://tauri.localhost` on Windows (we enable `useHttpsScheme=true`
    // in tauri.conf.json). Both must be in the backend's CORS allowlist or every
    // fetch from the UI is rejected. Set this unconditionally — a stale value in
    // the user's shell or a `.env` file shadowing the default would silently
    // break the app with cryptic CORS errors. In `tauri dev` the webview also
    // loads from the rsbuild dev server at http://localhost:3000 (see
    // tauri.conf.json `devUrl`), so that origin must be allowed too.
    #[cfg(debug_assertions)]
    let cors_origins = "tauri://localhost,https://tauri.localhost,http://localhost:3000";
    #[cfg(not(debug_assertions))]
    let cors_origins = "tauri://localhost,https://tauri.localhost";
    command.env("CORS_ORIGINS", cors_origins);

    // Resolve OS-conventional per-user dirs via Tauri (driven by the bundle
    // identifier in `tauri.conf.json`). These live outside the install prefix
    // so they survive reinstalls — same convention as Chrome/VSCode.
    //   macOS  : ~/Library/{Application Support, Logs, Caches}/com.intel.geti
    //   Windows: %APPDATA%\com.intel.geti\{,logs}, %LOCALAPPDATA%\com.intel.geti
    //   Linux  : ~/.local/share, ~/.local/state, ~/.cache (each /com.intel.geti)
    let resolver = app.path();
    set_env_from_dir(command, "DATA_DIR", resolver.app_local_data_dir().ok());
    set_env_from_dir(command, "LOG_DIR", resolver.app_log_dir().ok());
    // Pin matplotlib's font/style cache to a stable per-user dir so it's built
    // once and reused across launches. Without this, matplotlib falls back to
    // a path inside the frozen `_internal/` (re-extracted on every launch),
    // forcing a full font-cache rebuild every start.
    set_env_from_dir(
        command,
        "MPLCONFIGDIR",
        resolver.app_cache_dir().ok().map(|d| d.join("matplotlib")),
    );
}

/// Sets `key` to `dir` on `command` (creating the directory if needed), unless
/// the inherited environment already provides one or `dir` is `None`.
fn set_env_from_dir(command: &mut Command, key: &str, dir: Option<PathBuf>) {
    if env::var_os(key).is_some() {
        return;
    }
    let Some(dir) = dir else { return };
    let _ = std::fs::create_dir_all(&dir);
    command.env(key, dir);
}
