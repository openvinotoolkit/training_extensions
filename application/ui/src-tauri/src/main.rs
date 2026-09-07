// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
#[cfg(windows)]
mod job;
#[cfg(windows)]
mod webview;

use std::path::PathBuf;
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tauri::{AppHandle, Manager, RunEvent, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_opener::OpenerExt;

use crate::backend::spawn_backend;

/// How often the monitor thread checks whether the backend is still alive.
const MONITOR_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// How long a graceful shutdown request is given before the backend is killed.
///
/// Generous on purpose: the ASGI lifespan shutdown has to finish in-flight
/// requests, stop worker processes and close (checkpoint) the SQLite database.
const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Label of the single application window (see `tauri.conf.json`).
pub const MAIN_WINDOW_LABEL: &str = "main";

/// Public issue tracker where users can report fatal backend failures.
///
/// Native Windows dialogs render their body as non-selectable plain text, so a
/// URL printed there can neither be clicked nor copied. Instead of relying on
/// the user to type it out, the fatal dialogs offer a button that opens this URL
/// in the default browser (see [`open_issue_tracker`]). Keep it in the
/// `opener:allow-open-url` allowlist in `capabilities/default.json`.
const ISSUE_TRACKER_URL: &str = "https://github.com/open-edge-platform/geti/issues";

/// Open the issue tracker in the user's default browser, logging on failure.
fn open_issue_tracker(app: &AppHandle) {
    if let Err(e) = app.opener().open_url(ISSUE_TRACKER_URL, None::<String>) {
        log::warn!("Failed to open issue tracker URL {ISSUE_TRACKER_URL:?}: {e}");
    }
}

/// Exit code the backend uses to signal a fatal, non-restartable data-migration
/// failure during an in-place upgrade (see
/// `application/backend/app/lifecycle.py:MIGRATION_FATAL_EXIT_CODE`). Before
/// exiting with this code the backend has already rolled its data back to the
/// previous version, so the *previous* release remains usable — the newer one
/// simply cannot run against the existing data.
const MIGRATION_FATAL_EXIT_CODE: i32 = 3;

/// Name of the machine-readable status file the backend writes into `DATA_DIR`
/// right before it exits with `MIGRATION_FATAL_EXIT_CODE`. Keep this name and
/// schema in sync with `application/backend/app/lifecycle.py:FATAL_STATUS_FILENAME`.
const FATAL_STATUS_FILENAME: &str = "fatal_status.json";

/// Per-user directory the **Windows** backend actually uses for both data and
/// logs.
///
/// The frozen Windows backend's PyInstaller runtime hook
/// (`application/backend/pyinstaller/windows/uwp.py`) unconditionally overrides
/// `DATA_DIR` *and* `LOG_DIR` to `%LOCALAPPDATA%\Intel\Geti`, ignoring whatever
/// the shell passed on the command line. If the shell resolved these paths from
/// Tauri's bundle identifier (`com.intel.geti`) instead, its dialogs would point
/// users at the wrong folder and `read_and_clear_fatal_status` would look for the
/// backend's status file in the wrong place. Keep this in sync with `uwp.py`.
#[cfg(windows)]
fn backend_app_data_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Intel").join("Geti"))
}

/// Resolve the directory the backend uses for persistent data, matching the
/// backend's own precedence: an explicit `DATA_DIR` override wins, then (on
/// Windows) the hard-coded `%LOCALAPPDATA%\Intel\Geti` from `uwp.py`, and finally
/// Tauri's identifier-derived `app_local_data_dir()`.
fn resolve_data_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DATA_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    if let Some(dir) = backend_app_data_dir() {
        return Some(dir);
    }
    app.path().app_local_data_dir().ok()
}

/// Resolve the directory the backend writes logs to, mirroring [`resolve_data_dir`]
/// (on Windows `uwp.py` points `LOG_DIR` at the same `Intel\Geti` folder).
fn resolve_log_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("LOG_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    if let Some(dir) = backend_app_data_dir() {
        return Some(dir);
    }
    app.path().app_log_dir().ok()
}

/// Structured description of a fatal backend startup failure, deserialized from
/// [`FATAL_STATUS_FILENAME`]. Unknown fields are ignored so the backend can
/// extend the schema without breaking older shells.
#[derive(Debug, Default, Deserialize)]
struct FatalStatus {
    /// Machine-readable failure category, e.g. `"migration"`.
    #[serde(default)]
    fatal: String,
    /// Absolute path of the pre-migration backup to restore, if one was taken.
    #[serde(default)]
    backup_path: Option<String>,
    /// Absolute path of the database file the backup should be restored to.
    #[serde(default)]
    database_path: Option<String>,
}

/// Read and remove the backend's fatal-status file from the resolved data
/// directory (see [`resolve_data_dir`]).
///
/// Returns `None` if the file is absent or unreadable. The file is always
/// deleted after a successful read so a stale status can't resurface on the next
/// launch; the backend also clears it on a healthy start as a second safeguard.
fn read_and_clear_fatal_status(app: &AppHandle) -> Option<FatalStatus> {
    let path = resolve_data_dir(app)?.join(FATAL_STATUS_FILENAME);
    let contents = std::fs::read_to_string(&path).ok()?;

    let status = serde_json::from_str::<FatalStatus>(&contents)
        .map_err(|e| log::warn!("Failed to parse fatal status file {path:?}: {e}"))
        .ok();

    if let Some(status) = &status {
        log::info!("Read fatal status file {path:?}: reason={:?}", status.fatal);
    }

    if let Err(e) = std::fs::remove_file(&path) {
        log::warn!("Failed to remove fatal status file {path:?}: {e}");
    }

    status
}

/// Shared handle used to tear the backend down and to tell the monitor thread
/// whether an exit was intentional (so it doesn't mistake a clean shutdown for a
/// crash).
#[derive(Clone, Default)]
struct BackendControl {
    /// The live child process.
    ///
    /// The `Child` handle is kept (rather than a bare PID) for the whole
    /// lifetime of the backend: as long as it is not dropped, Windows/Linux
    /// cannot recycle that PID, so signalling or killing "the backend" can never
    /// hit an unrelated process that happens to have inherited the number.
    child: Arc<Mutex<Option<Child>>>,
    /// Windows job object keeping the backend tree bound to this process.
    #[cfg(windows)]
    job: Arc<Mutex<Option<job::JobHandle>>>,
    /// Set before we deliberately terminate the backend during app shutdown.
    shutting_down: Arc<AtomicBool>,
}

/// Block until `child` exits or `timeout` elapses. Returns the exit code when
/// the process terminated in time.
fn wait_for_exit(child: &mut Child, timeout: Duration) -> Option<Option<i32>> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status.code()),
            // Already reaped or not waitable: treat as gone, nothing left to kill.
            Err(_) => return Some(None),
            Ok(None) => {}
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(MONITOR_POLL_INTERVAL);
    }
}

/// Hard-kill a process and all its descendants by PID. Last resort only.
///
/// - **Windows**: `taskkill /F /T /PID` terminates the entire process tree.
/// - **Unix**: sends `SIGKILL` to the process group (`kill -- -<pid>`). The
///   backend is spawned as its own process-group leader (see `backend.rs`), so
///   all of its multiprocessing workers are included.
fn kill_process_tree(pid: u32) {
    #[cfg(windows)]
    {
        use std::process::Command;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }

    #[cfg(unix)]
    {
        use std::process::Command;
        // kill -- -PID sends the signal to the whole process group.
        let _ = Command::new("kill")
            .args(["-9", "--", &format!("-{pid}")])
            .output();
    }
}

/// Politely ask the backend tree to stop.
///
/// Windows gets a targeted `CTRL_BREAK_EVENT` (the backend runs in its own
/// console process group); Unix gets `SIGTERM` on the process group. Returns
/// `true` when the request was delivered.
fn request_graceful_stop(pid: u32) -> bool {
    #[cfg(windows)]
    {
        job::send_ctrl_break(pid)
    }

    #[cfg(unix)]
    {
        use std::process::Command;
        Command::new("kill")
            .args(["-TERM", "--", &format!("-{pid}")])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// Stop the backend as cleanly as possible.
///
/// A hard `taskkill /F /T` (the previous behaviour) leaves the SQLite WAL and
/// journal files behind and can truncate an in-flight migration or training
/// checkpoint, which then shows up as instability on the *next* launch. So the
/// backend is first asked to shut down, and only killed if it does not comply
/// within [`GRACEFUL_SHUTDOWN_TIMEOUT`].
fn terminate_backend(child: &mut Child) {
    let pid = child.id();

    if request_graceful_stop(pid) {
        if let Some(code) = wait_for_exit(child, GRACEFUL_SHUTDOWN_TIMEOUT) {
            log::info!("⛔ Backend terminated gracefully (exit code {code:?})");
            return;
        }
        log::warn!(
            "Backend did not stop within {}s, forcing termination",
            GRACEFUL_SHUTDOWN_TIMEOUT.as_secs()
        );
    }

    kill_process_tree(pid);
    let _ = child.kill();
    let _ = child.wait();
    log::info!("⛔ Backend terminated (forced)");
}

/// Deliberately terminate the backend (app shutdown). Marks the exit as intended
/// so the monitor thread stays silent instead of showing a crash dialog.
///
/// `reason` identifies the code path that triggered the shutdown; it is logged
/// so a session that ends unexpectedly can be traced back to the window being
/// closed, the runtime exiting, or a fatal backend failure.
fn shutdown_backend(control: &BackendControl, reason: &str) {
    // Idempotent: `CloseRequested` and `RunEvent::Exit` both call in.
    if control.shutting_down.swap(true, Ordering::SeqCst) {
        log::debug!("⛔ Backend shutdown already in progress (reason: {reason})");
        return;
    }

    log::info!("⛔ Shutting down the backend (reason: {reason})");
    if let Some(mut child) = control.child.lock().unwrap().take() {
        terminate_backend(&mut child);
    }

    // Releasing the job handle kills anything that somehow survived.
    #[cfg(windows)]
    drop(control.job.lock().unwrap().take());
}

/// Resolve the per-user log directory as a display string for user-facing
/// messages, falling back to a generic phrase if it can't be resolved.
fn log_dir_hint(app: &AppHandle) -> String {
    resolve_log_dir(app)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "the application log directory".to_string())
}

/// Detailed message shown when the backend aborts an in-place upgrade because
/// the data migration failed (exit code 3). When the backend provided a
/// [`FatalStatus`] with a backup path, the exact recovery instructions are
/// included so the user can restore their data by hand if needed.
fn show_migration_failure_dialog(app: &AppHandle, status: Option<&FatalStatus>) {
    let log_dir = log_dir_hint(app);

    // Assemble the body from independent paragraphs and join them with a single
    // blank line. Building it this way (instead of interpolating optional
    // fragments into one big format string) guarantees no stray empty line is
    // left behind when the recovery paragraph is absent.
    let mut paragraphs: Vec<String> = vec![
        "Geti tried to upgrade your data to this newer version, but the upgrade did not \
succeed."
            .to_string(),
        "The newer version of Geti cannot run with your existing data and will now close."
            .to_string(),
    ];

    if let Some(backup_path) = status.and_then(|s| s.backup_path.as_deref()) {
        let db_target = status
            .and_then(|s| s.database_path.as_deref())
            .map(|db| format!("'{db}' (the original database file)"))
            .unwrap_or_else(|| "the original database file".to_string());
        paragraphs.push(format!(
            "To recover, restore the pre-migration database backup: rename the backup file \
'{backup_path}' back to {db_target}, overwriting the partially migrated database. After \
restoring the backup, downgrade the application to the previous version."
        ));
    }

    paragraphs.push(format!("Logs are available at:\n  {log_dir}"));
    paragraphs.push(
        "If the problem persists, you can report it on our issue tracker and attach the log \
files."
            .to_string(),
    );

    let message = paragraphs.join("\n\n");

    let report = app
        .dialog()
        .message(message)
        .title("Geti upgrade failed")
        .kind(MessageDialogKind::Error)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Report issue".to_string(),
            "Close".to_string(),
        ))
        .blocking_show();

    if report {
        open_issue_tracker(app);
    }
}

/// Generic message shown when the backend stops unexpectedly for any other
/// reason, so the UI never just hangs with a dead backend.
fn show_backend_crash_dialog(app: &AppHandle, code: Option<i32>) {
    let log_dir = log_dir_hint(app);
    let code_str = code
        .map(|c| c.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!(
        "The Geti backend stopped unexpectedly (exit code {code_str}) and the application \
will now close.\n\n\
Please review the logs for details:\n  {log_dir}\n\n\
If this keeps happening, you can report it on our issue tracker and attach the log files."
    );

    let report = app
        .dialog()
        .message(message)
        .title("Geti stopped unexpectedly")
        .kind(MessageDialogKind::Error)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Report issue".to_string(),
            "Close".to_string(),
        ))
        .blocking_show();

    if report {
        open_issue_tracker(app);
    }
}

/// Wait for the backend to exit and react to unsolicited terminations. Runs on a
/// dedicated thread so it can call the *blocking* dialog API (which must not run
/// on the main thread).
///
/// The child handle lives in [`BackendControl`] and is polled (rather than
/// blocked on with `wait()`) so that ownership stays with the control struct —
/// that is what lets the shutdown path signal a PID that is guaranteed not to
/// have been recycled by the OS.
fn monitor_backend(app: AppHandle, control: BackendControl) {
    let code = loop {
        // A deliberate shutdown (window closed / app quit) already took the
        // child — the exit is expected, so stay silent.
        if control.shutting_down.load(Ordering::SeqCst) {
            return;
        }

        let status = {
            let mut guard = control.child.lock().unwrap();
            let Some(child) = guard.as_mut() else { return };
            child.try_wait()
        };

        match status {
            Ok(Some(status)) => break status.code(),
            Ok(None) => std::thread::sleep(MONITOR_POLL_INTERVAL),
            Err(e) => {
                log::warn!("Failed to poll the backend process: {e}");
                break None;
            }
        }
    };

    // Lost the race against a shutdown that started while we were polling.
    if control.shutting_down.swap(true, Ordering::SeqCst) {
        return;
    }

    // The backend is gone: release our handle (and the job object) so nothing
    // later signals a dead — possibly recycled — PID.
    drop(control.child.lock().unwrap().take());
    #[cfg(windows)]
    drop(control.job.lock().unwrap().take());

    log::warn!("Backend exited unexpectedly (code {code:?})");

    match code {
        Some(MIGRATION_FATAL_EXIT_CODE) => {
            log::error!("Backend reported a fatal upgrade/migration failure (exit code 3)");
            // The backend drops a status file into DATA_DIR (== app_local_data_dir)
            // describing the failure and, crucially, where the pre-migration backup
            // lives. Read it so the dialog can show the user the exact path.
            let status = read_and_clear_fatal_status(&app);
            show_migration_failure_dialog(&app, status.as_ref());
            app.exit(MIGRATION_FATAL_EXIT_CODE);
        }
        other => {
            show_backend_crash_dialog(&app, other);
            app.exit(other.unwrap_or(1));
        }
    }
}

fn main() {
    let control = BackendControl::default();

    let app = tauri::Builder::default()
        // Must be the first plugin registered: a second launch has to bail out
        // *before* `setup` spawns another backend. Two side-cars would fight
        // over the same TCP port and the same SQLite database in
        // %LOCALAPPDATA%\Intel\Geti, which shows up as random failures in
        // whichever instance loses the race.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            log::info!("▶ Second instance launched, focusing the existing window");
            if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_log::Builder::default().build())
        .setup({
            let control = control.clone();
            move |app| {
                // Logged up front so a mid-session WebView2 failure can be
                // correlated with an Evergreen runtime update: if this version
                // differs from the previous launch, the runtime serviced itself
                // underneath the app.
                #[cfg(windows)]
                webview::log_runtime_version();

                let sidecar = spawn_backend(app.handle()).expect("Failed to spawn python backend");
                // Keep the child handle — and the job object binding the whole
                // backend tree to our lifetime — in the shared control, then let
                // a monitor thread watch for crashes and failed upgrades
                // (exit code 3).
                #[cfg(windows)]
                {
                    *control.job.lock().unwrap() = sidecar.job;
                }
                *control.child.lock().unwrap() = Some(sidecar.child);

                // Recover from WebView2 renderer/browser-process failures
                // instead of letting them take the whole application down.
                #[cfg(windows)]
                if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
                    webview::attach_process_failed_handler(&window);
                }

                let app_handle = app.handle().clone();
                let monitor_control = control.clone();
                std::thread::spawn(move || monitor_backend(app_handle, monitor_control));
                Ok(())
            }
        })
        // Geti is a single-window utility app, so closing the main window
        // should quit the whole process (default macOS behaviour is to keep
        // the app alive in the dock, which leaks the backend side-car).
        .on_window_event({
            let control = control.clone();
            move |window, event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    // Prevent the default close so we can shut down gracefully.
                    // Destroying the window first lets the WebView2 / Chromium
                    // widget tear down cleanly before the process exits,
                    // avoiding the "Failed to unregister class
                    // Chrome_WidgetWin_0" error on Windows. It also means the
                    // user does not stare at a frozen window while the backend
                    // takes its (bounded) time to stop.
                    api.prevent_close();

                    let handle = window.app_handle().clone();
                    if let Err(e) = window.destroy() {
                        log::warn!("Failed to destroy window during shutdown: {e}");
                    }

                    // Stop the backend *before* exiting so worker processes
                    // cannot outlive the UI — even if RunEvent::Exit is
                    // short-circuited by exit(0).
                    shutdown_backend(&control, "window close requested");

                    handle.exit(0);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![])
        .build(tauri::generate_context!())
        .expect("error building Tauri");

    // Belt-and-suspenders: also handle RunEvent::Exit for cases where the app
    // exits without going through the CloseRequested path (e.g. Cmd+Q on
    // macOS, or programmatic shutdown).
    let exit_control = control.clone();
    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            shutdown_backend(&exit_control, "runtime exit event");
        }
    });
}
