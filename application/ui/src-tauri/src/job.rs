// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//! Windows-only process containment and graceful termination for the side-car.
//!
//! Two problems are solved here:
//!
//! 1. **Orphaned workers.** The Python backend spawns multiprocessing workers
//!    for training/export. If the shell crashes (or is killed from Task
//!    Manager) those workers keep running, keep the SQLite database open and
//!    keep the TCP port bound, so the next launch misbehaves. Assigning the
//!    side-car to a [Job Object] with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
//!    makes the kernel tear down the whole tree the moment our process exits —
//!    no cooperation from us required.
//!
//! 2. **Unclean shutdown.** `taskkill /F /T` hard-kills the backend, leaving
//!    the SQLite WAL/journal behind and truncating in-flight migrations or
//!    checkpoints. [`signal_shutdown_event`] instead asks the backend to stop,
//!    so it can run its ASGI lifespan shutdown and close the database properly;
//!    the hard kill stays only as a last-resort fallback.
//!
//! # Why a named event rather than Ctrl+Break
//!
//! The obvious way to stop a console child is `GenerateConsoleCtrlEvent`, but it
//! only reaches processes attached to the *caller's* console. The shell is a GUI
//! (`windows_subsystem = "windows"`) process with no console, and the release
//! side-car is spawned with `CREATE_NO_WINDOW`, which per [MSDN] means "the
//! console handle for the application is not set". With no console to borrow,
//! `AttachConsole` fails and every shutdown would silently fall through to the
//! forced kill — exactly the behaviour this module exists to avoid.
//!
//! So the primary channel is a named Win32 event ([`signal_shutdown_event`]),
//! which is completely independent of consoles: the backend creates it at
//! startup and waits on it in a background thread. [`send_ctrl_break`] is kept
//! only as a fallback for the console-attached `tauri dev` case and for older
//! backends that do not create the event yet.
//!
//! [Job Object]: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
//! [MSDN]: https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags

use std::os::windows::io::AsRawHandle;
use std::process::Child;

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Console::{
    AttachConsole, FreeConsole, GenerateConsoleCtrlEvent, GetConsoleWindow, SetConsoleCtrlHandler,
    CTRL_BREAK_EVENT,
};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows::Win32::System::Threading::{OpenEventW, SetEvent, EVENT_MODIFY_STATE};

/// Prefix of the per-process shutdown event the backend creates.
///
/// The full name is `Local\<prefix><pid>`. Keep in sync with
/// `SHUTDOWN_EVENT_PREFIX` in `application/backend/app/main.py`.
const SHUTDOWN_EVENT_PREFIX: &str = "geti-backend-shutdown-";

/// Owned handle to a job object configured to kill every contained process when
/// the handle is dropped (i.e. when this process exits, normally or not).
pub struct JobHandle(HANDLE);

// A job handle is just a kernel handle; it is safe to move across threads and
// the only operation we perform on it from another thread is `CloseHandle`.
unsafe impl Send for JobHandle {}
unsafe impl Sync for JobHandle {}

impl Drop for JobHandle {
    fn drop(&mut self) {
        // Closing the last handle to the job terminates all contained
        // processes, because of JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// Create an anonymous "kill on close" job object.
fn create_kill_on_close_job() -> windows::core::Result<JobHandle> {
    unsafe {
        let job = CreateJobObjectW(None, PCWSTR::null())?;
        let handle = JobHandle(job);

        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )?;

        Ok(handle)
    }
}

/// Put `child` (and every process it spawns) into a kill-on-close job object.
///
/// Returns the job handle, which must be kept alive for as long as the backend
/// should be allowed to run. Returns `None` when the job could not be created
/// or assigned — the app stays functional, it just loses the safety net.
pub fn contain(child: &Child) -> Option<JobHandle> {
    let job = match create_kill_on_close_job() {
        Ok(job) => job,
        Err(e) => {
            log::warn!("Failed to create job object for the backend side-car: {e}");
            return None;
        }
    };

    let process = HANDLE(child.as_raw_handle());
    if let Err(e) = unsafe { AssignProcessToJobObject(job.0, process) } {
        log::warn!("Failed to assign the backend side-car to a job object: {e}");
        return None;
    }

    log::info!("▶ Backend side-car contained in a kill-on-close job object");
    Some(job)
}

/// Ask the side-car to shut down gracefully by signalling its named event.
///
/// The backend creates a manual-reset event called
/// `Local\geti-backend-shutdown-<pid>` during startup and waits on it in a
/// background thread; setting it makes the backend run its normal shutdown path.
/// The other half of this contract lives in
/// `application/backend/pyinstaller/windows/shutdown.py` — keep the name format
/// in sync with `SHUTDOWN_EVENT_PREFIX` there.
///
/// The `Local\` namespace is per-session, so the event is visible to every
/// process in the user's session and to nothing outside it. The MSIX package is
/// full-trust, so it is not AppContainer-isolated and named-object access works
/// normally.
///
/// Returns `true` when the event was found and set.
pub fn signal_shutdown_event(pid: u32) -> bool {
    let name: Vec<u16> = format!("Local\\{SHUTDOWN_EVENT_PREFIX}{pid}\0")
        .encode_utf16()
        .collect();

    unsafe {
        let handle = match OpenEventW(EVENT_MODIFY_STATE, false, PCWSTR(name.as_ptr())) {
            Ok(handle) => handle,
            Err(e) => {
                // Expected when the backend predates this mechanism, or has not
                // finished starting up yet. The caller falls back to Ctrl+Break.
                log::warn!("No shutdown event for the backend (pid {pid}): {e}");
                return false;
            }
        };

        let signalled = SetEvent(handle);
        if let Err(e) = &signalled {
            log::warn!("Failed to set the backend shutdown event (pid {pid}): {e}");
        }
        let _ = CloseHandle(handle);

        signalled.is_ok()
    }
}

/// Ask the side-car to shut down gracefully by sending it `CTRL_BREAK_EVENT`.
///
/// **Fallback only** — see the module docs. This works in `tauri dev`, where the
/// shell owns a console the child inherits, but not for the release side-car
/// spawned with `CREATE_NO_WINDOW`, which has no console to attach to. Prefer
/// [`signal_shutdown_event`].
///
/// The backend is spawned with `CREATE_NEW_PROCESS_GROUP` (see `backend.rs`), so
/// it is its own console process group and can be signalled by PID.
///
/// `CTRL_C_EVENT` is deliberately not used: `CREATE_NEW_PROCESS_GROUP`
/// implicitly disables Ctrl+C handling for the new group, so only Ctrl+Break is
/// deliverable.
///
/// Returns `true` when the event was successfully generated.
pub fn send_ctrl_break(pid: u32) -> bool {
    unsafe {
        // In a console build (`tauri dev`) the child already shares our console,
        // so it can be signalled directly — and we must not detach ourselves.
        let owns_console = !GetConsoleWindow().is_invalid();
        if owns_console {
            let _ = SetConsoleCtrlHandler(None, true);
            let sent = GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid);
            let _ = SetConsoleCtrlHandler(None, false);
            if let Err(e) = &sent {
                log::warn!("Failed to send CTRL_BREAK to the backend (pid {pid}): {e}");
            }
            return sent.is_ok();
        }

        // Release builds are GUI (`windows_subsystem = "windows"`) processes
        // with no console at all, so borrow the child's for the duration of the
        // call. Detaching first is harmless and expected to fail when we own
        // nothing.
        let _ = FreeConsole();

        if let Err(e) = AttachConsole(pid) {
            log::warn!("Could not attach to backend console (pid {pid}): {e}");
            return false;
        }

        // While attached we would receive the event ourselves; ignore it so the
        // shell is not taken down together with the backend.
        let _ = SetConsoleCtrlHandler(None, true);

        let sent = GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid);
        if let Err(e) = &sent {
            log::warn!("Failed to send CTRL_BREAK to the backend (pid {pid}): {e}");
        }

        let _ = SetConsoleCtrlHandler(None, false);
        let _ = FreeConsole();

        sent.is_ok()
    }
}
