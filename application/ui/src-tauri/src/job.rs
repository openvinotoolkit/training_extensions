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
//!    checkpoints. [`send_ctrl_break`] instead delivers a console control event
//!    so the backend can run its ASGI lifespan shutdown and close the database
//!    properly; the hard kill stays only as a last-resort fallback.
//!
//! [Job Object]: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects

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

/// Ask the side-car to shut down gracefully by sending it `CTRL_BREAK_EVENT`.
///
/// The backend is spawned with `CREATE_NEW_PROCESS_GROUP` (see `backend.rs`), so
/// it is its own console process group and can be signalled by PID. Because the
/// shell is a GUI (`windows_subsystem = "windows"`) process without a console,
/// we temporarily attach to the child's console to be allowed to signal it.
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
