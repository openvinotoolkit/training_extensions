// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Default browser-build timeouts for the Segment Anything ONNX worker.
// Image embeddings come from the backend, so the worker only ever downloads and
// runs the decoder, which is far lighter than the encoder used to be.
//
// Tauri overrides these via `sam-timeouts.tauri.ts` (resolved by the
// `.tauri.*` extension list in rsbuild.config.ts) because WKWebView/WebView2
// either do not support WebGPU or fall back to CPU.

export const SAM_DECODER_TIMEOUT_MS = 20_000;
export const SAM_ASSET_DOWNLOAD_STALL_TIMEOUT_MS = 30_000;
export const SAM_WORKER_BUILD_TIMEOUT_MS = 30_000;
export const SAM_WORKER_INIT_TIMEOUT_MS = 60_000;

// How long an unobserved encoding (a 4 MiB Float32 tensor per image) is kept in
// the query cache before being garbage-collected. Short on purpose: encodings
// are heavy, so we favor memory over re-fetching them.
export const SAM_ENCODING_GC_TIME_MS = 60_000;
