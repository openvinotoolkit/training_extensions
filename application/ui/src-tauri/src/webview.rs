// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//! WebView2 resilience (Windows only).
//!
//! The app was observed exiting at random with
//! `WebView2 error: HRESULT(0x8007139F)` ("The group or resource is not in the
//! correct state…") and `REGDB_E_CLASSNOTREG` ("Class not registered"). That is
//! the signature of the **Evergreen WebView2 Runtime being serviced underneath a
//! running app**: the old runtime's COM classes are unregistered, the browser
//! process is torn down and every subsequent call on `ICoreWebView2` fails. A
//! GPU/renderer crash produces the same symptom.
//!
//! Tauri does not surface WebView2's `ProcessFailed` event, so such a failure
//! used to be fatal. Here we subscribe to it directly and recover:
//!
//! * renderer / frame-renderer gone or unresponsive → reload the page;
//! * browser process gone → the `ICoreWebView2` is unusable, so the whole
//!   window is recreated from its `tauri.conf.json` definition.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tauri::{AppHandle, Manager, WebviewWindow, WebviewWindowBuilder};
use webview2_com::Microsoft::Web::WebView2::Win32::{
    GetAvailableCoreWebView2BrowserVersionString, ICoreWebView2,
    ICoreWebView2ProcessFailedEventArgs, COREWEBVIEW2_PROCESS_FAILED_KIND,
    COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED,
    COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED,
    COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED,
    COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE,
};
use webview2_com::ProcessFailedEventHandler;
use windows::core::{PCWSTR, PWSTR};
use windows::Win32::System::Com::CoTaskMemFree;

use crate::MAIN_WINDOW_LABEL;

/// Log the installed Evergreen WebView2 Runtime version.
///
/// Recorded once at startup so a mid-session `ProcessFailed` can be correlated
/// with a runtime update: comparing this value across two consecutive launches
/// tells you immediately whether Evergreen serviced itself under the app.
pub fn log_runtime_version() {
    unsafe {
        let mut version = PWSTR::null();
        match GetAvailableCoreWebView2BrowserVersionString(PCWSTR::null(), &mut version) {
            Ok(()) if !version.is_null() => {
                let value = version.to_string().unwrap_or_default();
                CoTaskMemFree(Some(version.as_ptr() as *const _));
                log::info!("▶ WebView2 runtime version: {value}");
            }
            Ok(()) => log::warn!("WebView2 runtime version unavailable (null version string)"),
            Err(e) => log::warn!("Failed to query the WebView2 runtime version: {e}"),
        }
    }
}

/// Human-readable name for a `COREWEBVIEW2_PROCESS_FAILED_KIND`.
///
/// Written as comparisons rather than a `match` because the WebView2 kinds are
/// generated newtype constants, not enum variants.
fn kind_name(kind: COREWEBVIEW2_PROCESS_FAILED_KIND) -> &'static str {
    if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED {
        "browser process exited"
    } else if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED {
        "render process exited"
    } else if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE {
        "render process unresponsive"
    } else if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED {
        "frame render process exited"
    } else {
        "other process exited"
    }
}

/// Read the failure kind from the event args, defaulting to "unknown" when the
/// out-parameter cannot be read.
fn failed_kind(
    args: Option<&ICoreWebView2ProcessFailedEventArgs>,
) -> COREWEBVIEW2_PROCESS_FAILED_KIND {
    let mut kind = COREWEBVIEW2_PROCESS_FAILED_KIND::default();
    if let Some(args) = args {
        if let Err(e) = unsafe { args.ProcessFailedKind(&mut kind) } {
            log::warn!("Could not read the WebView2 process failure kind: {e}");
        }
    }
    kind
}

/// Subscribe to `ICoreWebView2::ProcessFailed` on `window` and recover from
/// failures instead of letting them kill the app.
///
/// Safe to call repeatedly (e.g. after recreating the window). Failures to
/// subscribe are logged and otherwise ignored — worst case we keep the previous
/// behaviour for that window.
pub fn attach_process_failed_handler(window: &WebviewWindow) {
    let app = window.app_handle().clone();
    // `ProcessFailed` can fire several times in quick succession while the
    // runtime tears itself down; only the first one should trigger recovery.
    let recovering = Arc::new(AtomicBool::new(false));

    let result = window.with_webview(move |platform_webview| {
        let controller = platform_webview.controller();
        let webview: ICoreWebView2 = match unsafe { controller.CoreWebView2() } {
            Ok(webview) => webview,
            Err(e) => {
                log::warn!("Could not obtain ICoreWebView2 to watch for process failures: {e}");
                return;
            }
        };

        let handler = ProcessFailedEventHandler::create(Box::new(
            move |_sender: Option<ICoreWebView2>,
                  args: Option<ICoreWebView2ProcessFailedEventArgs>| {
                let kind = failed_kind(args.as_ref());
                log::error!("WebView2 process failed: {} ({kind:?})", kind_name(kind));

                if recovering.swap(true, Ordering::SeqCst) {
                    log::warn!("WebView2 recovery already in progress, ignoring event");
                    return Ok(());
                }

                let app = app.clone();
                let recovering = recovering.clone();
                // The event arrives on the UI thread inside a COM callback;
                // defer the actual recovery so the runtime can finish
                // dispatching before we touch (or destroy) the webview.
                let dispatch = app.clone().run_on_main_thread(move || {
                    recover(&app, kind);
                    recovering.store(false, Ordering::SeqCst);
                });
                if let Err(e) = dispatch {
                    log::error!("Failed to schedule WebView2 recovery: {e}");
                }
                Ok(())
            },
        ));

        let mut token = Default::default();
        if let Err(e) = unsafe { webview.add_ProcessFailed(&handler, &mut token) } {
            log::warn!("Failed to subscribe to WebView2 ProcessFailed: {e}");
        } else {
            log::info!("▶ Watching WebView2 for process failures");
        }
    });

    if let Err(e) = result {
        log::warn!("Failed to access the platform webview: {e}");
    }
}

/// Recover from a WebView2 process failure.
fn recover(app: &AppHandle, kind: COREWEBVIEW2_PROCESS_FAILED_KIND) {
    let browser_gone = kind == COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED;

    if !browser_gone {
        // Only a renderer died: the controller is still usable, so a reload
        // brings the UI back without losing the window (or the backend).
        if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
            log::info!("↻ Reloading the webview after a renderer failure");
            if let Err(e) = window.eval("window.location.reload()") {
                log::warn!("Reload failed, recreating the window instead: {e}");
                recreate_main_window(app);
            }
            return;
        }
    }

    log::info!("↻ Recreating the main window after a WebView2 browser-process failure");
    recreate_main_window(app);
}

/// Destroy and rebuild the main window from its `tauri.conf.json` definition.
///
/// After a browser-process failure the existing `ICoreWebView2` can no longer be
/// used for anything, so a brand new controller (and window) is the only way
/// back. The backend side-car is untouched and keeps running.
fn recreate_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        if let Err(e) = window.destroy() {
            log::warn!("Failed to destroy the failed window: {e}");
        }
    }

    let config = app
        .config()
        .app
        .windows
        .iter()
        .find(|w| w.label == MAIN_WINDOW_LABEL)
        .cloned();

    let Some(config) = config else {
        log::error!("No '{MAIN_WINDOW_LABEL}' window in the config; cannot recover");
        return;
    };

    match WebviewWindowBuilder::from_config(app, &config).and_then(|b| b.build()) {
        Ok(window) => {
            log::info!("✔ Main window recreated after WebView2 failure");
            attach_process_failed_handler(&window);
        }
        Err(e) => log::error!("Failed to recreate the main window: {e}"),
    }
}
