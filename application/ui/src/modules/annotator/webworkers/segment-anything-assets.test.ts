// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { env } from 'onnxruntime-web';

vi.mock('onnxruntime-web', () => ({ env: { wasm: {} } }));
vi.mock('@geti-ui/smart-tools/utils', () => ({ setOpenCVSourceUrl: vi.fn() }));
vi.mock('@geti-ui/smart-tools/segment-anything', () => ({
    SegmentAnythingModels: { decoder: '/static/assets/decoder.onnx' },
}));

const ORIGIN = self.location.origin;
const OPENCV_URL = `${ORIGIN}/opencv/opencv.js`;
const ORT_WASM_URL = `${ORIGIN}/ort/ort-wasm-simd-threaded.jsep.wasm`;
const DECODER_URL = `${ORIGIN}/static/assets/decoder.onnx`;
const SIZES: Record<string, number> = { [OPENCV_URL]: 3, [ORT_WASM_URL]: 5, [DECODER_URL]: 7 };
const STALL_TIMEOUT_MS = 30_000;

const toUrl = (input: RequestInfo | URL) =>
    typeof input === 'string' ? input : 'url' in input ? input.url : input.href;

const respond = (url: string, signal: AbortSignal | null | undefined, stall: boolean) => {
    const body = new ReadableStream<Uint8Array>({
        start(controller) {
            signal?.addEventListener('abort', () => controller.error(signal.reason));
            controller.enqueue(new Uint8Array(SIZES[url]));

            if (!stall) {
                controller.close();
            }
        },
    });

    return new Response(body);
};

const mockNetwork = (stallingUrl?: string) => {
    const networkFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) =>
        respond(toUrl(input), init?.signal, toUrl(input) === stallingUrl)
    );

    vi.stubGlobal('fetch', networkFetch);

    return networkFetch;
};

const loadModule = async () => {
    vi.resetModules();

    return import('./segment-anything-assets');
};

describe('segment anything assets', () => {
    beforeEach(() => {
        vi.stubEnv('ASSET_PREFIX', '');
        env.wasm.wasmBinary = undefined;
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
        vi.unstubAllEnvs();
    });

    it('serves prefetched assets from memory until they are released', async () => {
        const networkFetch = mockNetwork();
        const { prefetchSegmentAnythingAssets, releaseSegmentAnythingAssets } = await loadModule();

        await prefetchSegmentAnythingAssets(STALL_TIMEOUT_MS);

        expect((env.wasm.wasmBinary as ArrayBuffer).byteLength).toBe(SIZES[ORT_WASM_URL]);

        networkFetch.mockClear();

        expect((await (await self.fetch(new URL(OPENCV_URL))).arrayBuffer()).byteLength).toBe(SIZES[OPENCV_URL]);
        expect((await (await self.fetch(DECODER_URL)).arrayBuffer()).byteLength).toBe(SIZES[DECODER_URL]);
        expect(networkFetch).not.toHaveBeenCalled();

        releaseSegmentAnythingAssets();
        await self.fetch(DECODER_URL);

        expect(env.wasm.wasmBinary).toBeUndefined();
        expect(networkFetch).toHaveBeenCalledTimes(1);
    });

    it('fails a download only after no data arrives for the stall timeout', async () => {
        vi.useFakeTimers();
        mockNetwork(ORT_WASM_URL);

        const { prefetchSegmentAnythingAssets } = await loadModule();
        const prefetch = prefetchSegmentAnythingAssets(STALL_TIMEOUT_MS);
        const rejection = expect(prefetch).rejects.toThrow(
            `No data received for 30s while downloading ${ORT_WASM_URL}`
        );

        await vi.advanceTimersByTimeAsync(STALL_TIMEOUT_MS - 1);
        expect(env.wasm.wasmBinary).toBeUndefined();

        await vi.advanceTimersByTimeAsync(1);
        await rejection;
    });
});
