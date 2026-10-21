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
//! used to be fatal. Here we subscribe to it directly and recover according to
//! what is still usable:
//!
//! * renderer / frame-renderer *exited* → a native reload starts a new
//!   renderer and repaints the UI in the same window;
//! * renderer *unresponsive* → the window is recreated. Reloading is useless
//!   here: `eval` only queues a script for a renderer that is not draining its
//!   task queue, so it reports success while the UI stays frozen;
//! * browser process gone → the `ICoreWebView2` is unusable, so the whole
//!   window is recreated from its `tauri.conf.json` definition;
//! * anything else (GPU, utility, sandbox-helper processes) → nothing, WebView2
//!   restarts those itself.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use tauri::{AppHandle, Manager, RunEvent, WebviewWindow, WebviewWindowBuilder, WindowEvent};
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

/// How many consecutive "renderer unresponsive" notifications are tolerated
/// before the window is recreated.
///
/// WebView2 re-raises `ProcessFailed` for as long as the renderer stays hung, so
/// counting them distinguishes a genuinely wedged renderer from a long
/// synchronous task (a big annotation render, say) that would recover on its own
/// — and whose window should not be thrown away.
const UNRESPONSIVE_STRIKES_BEFORE_RECREATE: u32 = 3;

/// Set while the main window is being destroyed so that it is rebuilt once the
/// runtime has released its label (see [`on_run_event`]).
static RECREATE_PENDING: AtomicBool = AtomicBool::new(false);

/// What can still be done with the webview after a given failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Recovery {
    /// WebView2 restarts the process itself; nothing for us to do.
    None,
    /// A fresh renderer is already in place, so a reload repaints the UI.
    Reload,
    /// The webview can no longer be driven; build a new window around a new
    /// controller.
    Recreate,
}

/// Map a failure kind to the recovery it needs.
fn recovery_for(kind: COREWEBVIEW2_PROCESS_FAILED_KIND) -> Recovery {
    if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED {
        // Every call on the existing ICoreWebView2 fails from here on.
        Recovery::Recreate
    } else if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE {
        // The renderer is alive but not running our script, so anything that
        // goes through it (`eval`, navigation) silently does nothing.
        Recovery::Recreate
    } else if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED
        || kind == COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED
    {
        Recovery::Reload
    } else {
        Recovery::None
    }
}

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
    // Consecutive "renderer unresponsive" notifications, see
    // [`UNRESPONSIVE_STRIKES_BEFORE_RECREATE`].
    let unresponsive_strikes = Arc::new(AtomicU32::new(0));

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
                let unresponsive =
                    kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE;
                log::error!("WebView2 process failed: {} ({kind:?})", kind_name(kind));

                // Give a hung renderer a few chances to come back before the
                // window (and the user's view state) is discarded.
                if unresponsive {
                    let strikes = unresponsive_strikes.fetch_add(1, Ordering::SeqCst) + 1;
                    if strikes < UNRESPONSIVE_STRIKES_BEFORE_RECREATE {
                        log::warn!(
                            "Renderer unresponsive \
                             ({strikes}/{UNRESPONSIVE_STRIKES_BEFORE_RECREATE}), waiting"
                        );
                        return Ok(());
                    }
                } else {
                    unresponsive_strikes.store(0, Ordering::SeqCst);
                }

                let recovery = recovery_for(kind);
                if recovery == Recovery::None {
                    log::info!("WebView2 restarts this process itself, no action needed");
                    return Ok(());
                }

                if recovering.swap(true, Ordering::SeqCst) {
                    log::warn!("WebView2 recovery already in progress, ignoring event");
                    return Ok(());
                }

                let app = app.clone();
                // Kept out of the deferred closure so the flag can still be
                // cleared if the dispatch itself fails.
                let recovering_guard = recovering.clone();
                let recovering = recovering.clone();
                let unresponsive_strikes = unresponsive_strikes.clone();
                // Already on the UI thread, so this runs inline; window
                // teardown is queued by the runtime and finishes later.
                let dispatch = app.clone().run_on_main_thread(move || {
                    recover(&app, recovery);
                    unresponsive_strikes.store(0, Ordering::SeqCst);
                    recovering.store(false, Ordering::SeqCst);
                });
                if let Err(e) = dispatch {
                    // Nothing was scheduled, so release the latch or recovery
                    // would be blocked for every later failure.
                    log::error!("Failed to schedule WebView2 recovery: {e}");
                    recovering_guard.store(false, Ordering::SeqCst);
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

/// Carry out the recovery decided by [`recovery_for`].
fn recover(app: &AppHandle, recovery: Recovery) {
    match recovery {
        // Never reached (filtered out in the handler), but keeps the match total.
        Recovery::None => {}
        Recovery::Reload => {
            // Must be a native reload: a script reload (`eval`) is silently
            // dropped because there is no renderer left to run it.
            if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
                log::info!("↻ Reloading the webview after a renderer failure");
                match window.reload() {
                    Ok(()) => return,
                    Err(e) => log::warn!("Reload failed, recreating the window instead: {e}"),
                }
            }
            recreate_main_window(app);
        }
        Recovery::Recreate => {
            log::info!("↻ Recreating the main window: the webview can no longer be driven");
            recreate_main_window(app);
        }
    }
}

/// Destroy and rebuild the main window from its `tauri.conf.json` definition.
///
/// After a browser-process failure the existing `ICoreWebView2` can no longer be
/// used for anything, so a brand new controller (and window) is the only way
/// back. The backend side-car is untouched and keeps running.
///
/// `destroy()` is asynchronous and the label stays registered until the window
/// is gone, so the replacement is built from [`on_run_event`] once the old
/// window reports `Destroyed`.
fn recreate_main_window(app: &AppHandle) {
    let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) else {
        build_main_window(app);
        return;
    };

    // ProcessFailed arrives in bursts; a repeat must not cancel the pending rebuild.
    if RECREATE_PENDING.swap(true, Ordering::SeqCst) {
        log::info!("Main window recreation already pending, ignoring");
        return;
    }
    if let Err(e) = window.destroy() {
        RECREATE_PENDING.store(false, Ordering::SeqCst);
        log::error!("Failed to destroy the failed window; cannot recover: {e}");
    }
}

/// Hook for the application run loop.
///
/// Builds the replacement window as soon as the old one is destroyed. Doing it
/// inside this callback, before the runtime notices it has no windows left,
/// also stops the app from treating the teardown as "last window closed" and
/// quitting.
pub fn on_run_event(app: &AppHandle, event: &RunEvent) {
    if let RunEvent::WindowEvent {
        label,
        event: WindowEvent::Destroyed,
        ..
    } = event
    {
        if label == MAIN_WINDOW_LABEL && RECREATE_PENDING.swap(false, Ordering::SeqCst) {
            build_main_window(app);
        }
    }
}

/// Build the main window from its `tauri.conf.json` definition and watch it for
/// WebView2 process failures.
fn build_main_window(app: &AppHandle) {
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
