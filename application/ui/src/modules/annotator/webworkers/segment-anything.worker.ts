// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildSegmentAnythingInstance } from '@geti-ui/smart-tools/segment-anything';
import { setOrtWasmPaths } from '@geti-ui/smart-tools/utils';
import { expose, proxy } from 'comlink';

import { ORT_WASM_ROOT, prefetchSegmentAnythingAssets, releaseSegmentAnythingAssets } from './segment-anything-assets';
import type { SegmentAnythingWorkerApi } from './segment-anything.worker.interface';

import './opencv-source';
import './ort-session-params';

// `@geti-ui/smart-tools` doesn't hardcode ONNX Runtime's wasm location; the
// consuming app must tell it where the `onnxruntime-web` wasm/mjs artifacts
// are served from. rsbuild copies them to `ort/` (see rsbuild.config.ts),
// but the app is commonly served behind `ASSET_PREFIX` (e.g. `/html` in the
// Docker/install.sh build — see application/backend/app/main.py's static
// mount), so the absolute URL must include that prefix or it 404s in prod.
// Must run before any SAM session is created.
setOrtWasmPaths(ORT_WASM_ROOT);

const WorkerApi: SegmentAnythingWorkerApi = {
    prefetchAssets: prefetchSegmentAnythingAssets,
    build: async () => {
        const instance = await buildSegmentAnythingInstance();

        return proxy(instance);
    },
    releaseAssets: releaseSegmentAnythingAssets,
};

expose(WorkerApi);
