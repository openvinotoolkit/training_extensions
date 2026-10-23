// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { setOpenCVSourceUrl } from '@geti-ui/smart-tools/utils';

// `@geti-ui/smart-tools` doesn't bundle or serve the compiled `opencv.js`
// binary (it's compiled separately and vendored at vendor/opencv/, copied to
// `dist/opencv/` by rsbuild — see rsbuild.config.ts `output.copy`). String
// sources passed to `setOpenCVSourceUrl` are resolved against the running
// app's origin (`location.origin`), NOT the configured asset prefix, so the
// prefix (e.g. `/html` in the Docker/install.sh production build — see
// application/backend/app/main.py's static mount) must be prepended
// ourselves, same as the ORT wasm path handling in segment-anything.worker.ts.
// Import this module (for its side effect) before using any OpenCV-backed
// tool: GrabCut, Intelligent Scissors, Watershed, SSIM, RITM, and Segment
// Anything's mask-to-polygon postprocessing.
export const OPENCV_SOURCE_PATH = `${process.env.ASSET_PREFIX}/opencv/opencv.js`;

setOpenCVSourceUrl(OPENCV_SOURCE_PATH);
