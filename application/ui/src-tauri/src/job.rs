// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//! Windows-only process containment for the side-car.
//!
//! The Python backend spawns multiprocessing workers for training/export. If the
//! shell exits — or crashes, or is killed from Task Manager — those workers keep
//! running, keep the SQLite database open and keep the TCP port bound, so the
//! next launch misbehaves.
//!
//! Assigning the side-car to a [Job Object] with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` makes the kernel tear the whole tree
//! down the moment the last handle to the job closes. That covers both the
//! orderly path (the shell drops the handle during shutdown) and the abrupt one
//! (the handle is closed by the kernel when the shell dies), and it costs a
//! single handle close — no helper process, no timeout, no cooperation from the
//! backend.
//!
//! [Job Object]: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects

use std::os::windows::io::AsRawHandle;
use std::process::Child;

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
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
