// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Tauri-specific SAM timeout budgets, picked up automatically via the
// `.tauri.*` resolver in rsbuild.config.ts. Tauri ships a system webview
// (WKWebView on macOS, WebView2 on Windows, WebKitGTK on Linux); WebGPU is
// either unavailable or behind flags, so onnxruntime-web falls back to CPU.
// Only the decoder still runs locally, but CPU decoding is slow enough that
// the browser budgets produce false positives on a healthy worker.

export const SAM_DECODER_TIMEOUT_MS = 30_000;
export const SAM_ASSET_DOWNLOAD_STALL_TIMEOUT_MS = 30_000;
export const SAM_WORKER_BUILD_TIMEOUT_MS = 30_000;
export const SAM_WORKER_INIT_TIMEOUT_MS = 60_000;

export const SAM_ENCODING_GC_TIME_MS = 60_000;
