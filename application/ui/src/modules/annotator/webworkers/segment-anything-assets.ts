// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SegmentAnythingModels } from '@geti-ui/smart-tools/segment-anything';
import { env } from 'onnxruntime-web';

import { OPENCV_SOURCE_PATH } from './opencv-source';

export const ORT_WASM_ROOT = `${process.env.ASSET_PREFIX}/ort/`;

const toAbsoluteUrl = (url: string | URL) => new URL(url, self.location.href).href;

const OPENCV_URL = toAbsoluteUrl(OPENCV_SOURCE_PATH);
const ORT_WASM_URL = toAbsoluteUrl(`${ORT_WASM_ROOT}ort-wasm-simd-threaded.jsep.wasm`);
const DECODER_URL = toAbsoluteUrl(SegmentAnythingModels.decoder);

const prefetchedAssets = new Map<string, ArrayBuffer>();
const networkFetch = self.fetch.bind(self);

// smart-tools fetches OpenCV and the decoder itself.
self.fetch = (input, init) => {
    const asset = prefetchedAssets.get(input instanceof Request ? input.url : toAbsoluteUrl(input));

    return asset === undefined ? networkFetch(input, init) : Promise.resolve(new Response(asset));
};

const download = async (url: string, stallTimeoutMs: number): Promise<ArrayBuffer> => {
    const controller = new AbortController();
    let stallTimer: ReturnType<typeof setTimeout> | undefined;

    const restartStallTimer = () => {
        clearTimeout(stallTimer);
        stallTimer = setTimeout(() => {
            controller.abort(new Error(`No data received for ${stallTimeoutMs / 1000}s while downloading ${url}`));
        }, stallTimeoutMs);
    };

    try {
        restartStallTimer();

        const response = await networkFetch(url, { signal: controller.signal });

        if (!response.ok || response.body === null) {
            throw new Error(`Failed to download ${url} (${response.status})`);
        }

        const restartStallTimerOnEveryChunk = new TransformStream<Uint8Array, Uint8Array>({
            transform: (chunk, stream) => {
                restartStallTimer();
                stream.enqueue(chunk);
            },
        });

        return await new Response(response.body.pipeThrough(restartStallTimerOnEveryChunk)).arrayBuffer();
    } catch (error) {
        throw controller.signal.aborted ? controller.signal.reason : error;
    } finally {
        clearTimeout(stallTimer);
    }
};

export const prefetchSegmentAnythingAssets = async (stallTimeoutMs: number) => {
    const [opencv, ortWasm, decoder] = await Promise.all(
        [OPENCV_URL, ORT_WASM_URL, DECODER_URL].map((url) => download(url, stallTimeoutMs))
    );

    prefetchedAssets.set(OPENCV_URL, opencv);
    prefetchedAssets.set(DECODER_URL, decoder);
    env.wasm.wasmBinary = ortWasm;
};

export const releaseSegmentAnythingAssets = () => {
    prefetchedAssets.clear();
    env.wasm.wasmBinary = undefined;
};
