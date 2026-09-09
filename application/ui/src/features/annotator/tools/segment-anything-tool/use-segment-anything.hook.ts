// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fetchClient } from '@/api';
import type { Media } from '@/api/types';
import { EncodingOutput, InvalidEncodingError, parseEncoding } from '@geti-ui/smart-tools/segment-anything';
import { queryOptions, skipToken, useQuery, type QueryKey } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';
import { useProject } from 'hooks/api/project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { getQueryKey } from '../../../../query-client/query-client';
import { isVideo, isVideoFrame } from '../../../../shared/media-item-utils';
import { isDetectionTask } from '../../../project/task-type-guards';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import type {
    SegmentAnythingWorkerApi,
    SegmentAnythingWorkerInstance,
} from '../../webworkers/segment-anything.worker.interface';
import { executeWithTimeout } from '../execute-with-timeout';
import { convertToolShapeToGetiShape } from '../utils';
import {
    SAM_DECODER_TIMEOUT_MS,
    SAM_ENCODING_GC_TIME_MS,
    SAM_WORKER_BUILD_TIMEOUT_MS,
    SAM_WORKER_INIT_TIMEOUT_MS,
} from './sam-timeouts';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

type SegmentAnythingRemoteInstance = Remote<SegmentAnythingWorkerInstance>;

// `gcTime: Infinity` is critical: the default 5-min gc would evict the worker
// entry whenever SAM is unmounted (switching tools/projects), causing the
// next mount to spawn a brand-new worker that re-downloads the ORT wasm +
// JSEP `.mjs` glue + the multi-MB SAM `.onnx` models. It would also leak the
// previous worker, since tanstack doesn't fire the abort signal on gc — only
// on cancellation of an in-flight query. Keeping the entry alive for the
// lifetime of the page reuses the same worker (and its resident sessions)
// across every mount.
const segmentAnythingWorkerQueryOptions = (enabled = true) =>
    queryOptions<{ worker: Worker; instance: SegmentAnythingRemoteInstance }>({
        queryKey: ['workers', 'SEGMENT_ANYTHING'],
        queryFn: async ({ signal }) => {
            const worker = new Worker(new URL('../../webworkers/segment-anything.worker', import.meta.url), {
                type: 'module',
            });
            // Terminate the worker if the query is cancelled (e.g. annotator unmounts)
            // before build/init resolve, so we don't leak the in-flight worker.
            //
            // CRITICAL: pass an arrow function, NOT `worker.terminate` directly.
            // addEventListener invokes the listener with `this = signal`, and
            // `Worker.prototype.terminate` requires `this` to be a Worker — it
            // throws "Illegal invocation" silently and the worker is never
            // killed. With React StrictMode double-mounting effects (or any
            // transient cancellation), every cycle leaked a fresh worker, each
            // of which independently re-fetched opencv (~1MB), the ort bundle,
            // ort wasm, and the SAM `.onnx` models. That was the dominant
            // source of the "ort/opencv fetched 6 times" symptom.
            signal.addEventListener('abort', () => worker.terminate(), { once: true });

            try {
                const samWorker = wrap<SegmentAnythingWorkerApi>(worker);
                const instance = await executeWithTimeout(
                    samWorker.build(),
                    'SAM worker build',
                    SAM_WORKER_BUILD_TIMEOUT_MS
                );

                // Only the decoder runs locally; image embeddings come from the backend.
                await executeWithTimeout(
                    instance.init('SEGMENT_ANYTHING_DECODER'),
                    'SAM worker init',
                    SAM_WORKER_INIT_TIMEOUT_MS
                );

                if (signal.aborted) {
                    throw signal.reason;
                }

                return { worker, instance };
            } catch (error) {
                worker.terminate();

                throw error;
            }
        },
        staleTime: Infinity,
        gcTime: Infinity,
        enabled,
    });

const getEncodingQueryParams = (mediaItem: Media) =>
    isVideoFrame(mediaItem) ? { frame_index: mediaItem.frame_number } : undefined;

// Prefix matching every media's encoding query, so callers can cancel them all at once.
export const SEGMENT_ANYTHING_ENCODING_QUERY_KEY_PREFIX: QueryKey = [
    'get',
    '/api/projects/{project_id}/dataset/media/{media_id}/embeddings',
];

const getSegmentAnythingEncodingQueryKey = (projectId: string, mediaItem: Media): QueryKey =>
    getQueryKey([
        'get',
        '/api/projects/{project_id}/dataset/media/{media_id}/embeddings',
        {
            params: {
                path: { project_id: projectId, media_id: mediaItem.id },
                query: getEncodingQueryParams(mediaItem),
            },
        },
    ]);

class EmbeddingRequestError extends Error {
    constructor(
        readonly status: number,
        statusText: string
    ) {
        super(`Could not fetch the image embedding (${status}${statusText ? ` ${statusText}` : ''}).`);

        this.name = 'EmbeddingRequestError';
    }
}

const segmentAnythingEncodingQueryOptions = (projectId: string, mediaItem: Media, enabled = true) =>
    queryOptions({
        queryKey: getSegmentAnythingEncodingQueryKey(projectId, mediaItem),
        queryFn: async ({ signal }) => {
            // No `executeWithTimeout` here: the query's abort signal is what cancels the
            // in-flight download when the user moves to another media item.
            const { data, response } = await fetchClient.GET(
                '/api/projects/{project_id}/dataset/media/{media_id}/embeddings',
                {
                    params: {
                        path: { project_id: projectId, media_id: mediaItem.id },
                        query: getEncodingQueryParams(mediaItem),
                    },
                    parseAs: 'arrayBuffer',
                    signal,
                }
            );

            if (!response.ok || data === undefined) {
                throw new EmbeddingRequestError(response.status, response.statusText);
            }

            return parseEncoding(data);
        },
        staleTime: Infinity,
        gcTime: SAM_ENCODING_GC_TIME_MS,
        retry: (failureCount, error) => {
            // A malformed payload is a contract violation; retrying cannot fix it.
            if (error instanceof InvalidEncodingError) {
                return false;
            }

            // Neither can a client error (unknown media item, missing frame index).
            // Server errors are transient though: the backend serialises SAM inference
            // and answers 503 while another request is running.
            if (error instanceof EmbeddingRequestError && error.status < 500) {
                return false;
            }

            return failureCount < 3;
        },
        enabled,
    });

export const useSegmentAnythingWorker = (enabled = true) => {
    return useQuery({
        ...segmentAnythingWorkerQueryOptions(enabled),
        select: (data) => data.instance,
    });
};

const useEncodingQuery = (projectId: string, mediaItem: Media | undefined, enabled = true) => {
    // A whole video has no frame to encode. The selected media item is always converted
    // to a video frame, but `nextMediaItem` comes straight from the dataset listing and
    // can still be the video itself; asking for its embedding would 400.
    const isEncodable = mediaItem !== undefined && !isVideo(mediaItem);

    return useQuery(
        isEncodable
            ? segmentAnythingEncodingQueryOptions(projectId, mediaItem, enabled)
            : {
                  queryKey: ['segment-anything-model', 'encoding', 'disabled'],
                  queryFn: skipToken,
              }
    );
};

const useDecoderOutputType = () => {
    const { data } = useProject();

    if (isDetectionTask(data.task.task_type)) {
        return 'rect';
    }

    return 'polygon';
};

const useDecodingFn = (model: SegmentAnythingRemoteInstance | undefined, encoding: EncodingOutput | undefined) => {
    const decoderOutput = useDecoderOutputType();

    // TODO: look into returning a new "decoder model" instance that already has the encoding data
    // stored in memory, to reduce  memory usage
    return async (points: InteractiveAnnotationPoint[]) => {
        if (points.length === 0) {
            return [];
        }

        if (model === undefined) {
            return [];
        }

        if (encoding === undefined) {
            return [];
        }

        const { shapes } = await executeWithTimeout(
            model.processDecoder(encoding, {
                points,
                boxes: [],
                // Decoding runs against the already-computed `encoding`; no image is needed.
                image: undefined,
                outputConfig: {
                    type: decoderOutput,
                },
            }),
            'SAM decoder',
            SAM_DECODER_TIMEOUT_MS
        );

        return shapes.map(convertToolShapeToGetiShape);
    };
};

type SegmentAnythingModelOptions = {
    nextMediaItem?: Media;
};

export const useSegmentAnythingModel = ({ nextMediaItem }: SegmentAnythingModelOptions = {}) => {
    const workerQuery = useSegmentAnythingWorker();
    const model = workerQuery.data;
    const hasWorkerError = workerQuery.isError;
    const isLoadingWorkers = workerQuery.isLoading;
    const projectId = useProjectIdentifier();

    const { mediaItem } = useSelectedMediaItem();

    // First we get the encoding for the CURRENT image
    const encodingQuery = useEncodingQuery(projectId, mediaItem);

    // At the same time we start prefetching the encoding for the NEXT image,
    // so when the user moves to the next media item the decoding will be faster.
    // We don't need to get the decoding query result for the next image, we just want to cache the encoding result.
    // The backend serialises SAM inference, so we hold off while the current one is in flight.
    useEncodingQuery(projectId, nextMediaItem, !encodingQuery.isFetching);

    const decodingQueryFn = useDecodingFn(model, encodingQuery.data);

    const isLoading = !hasWorkerError && (isLoadingWorkers || encodingQuery.isLoading);
    const isError = hasWorkerError || encodingQuery.isError;
    const error = workerQuery.error ?? encodingQuery.error;

    return {
        isLoading,
        isError,
        error,
        decodingQueryFn,
    };
};
