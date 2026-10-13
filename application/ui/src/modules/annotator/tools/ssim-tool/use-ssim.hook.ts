// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from 'react';

import { i18n } from '@/i18n';
import { roiFromImage } from '@geti-ui/smart-tools/utils';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import type { Shape } from '../../../../shared/types';
import type { SSIMWorkerApi, SSIMWorkerInstance } from '../../webworkers/ssim-worker.interface';
import { executeWithTimeout } from '../execute-with-timeout';
import {
    convertRectToShape,
    convertToolMatchesToGetiMatches,
    filterSSIMResults,
    getExistingRects,
    guessNumberOfItemsThreshold,
    toToolRunSSIMProps,
    type RunSSIMProps,
    type SSIMMatch,
    type SSIMShapeType,
} from './utils';

// Building the worker downloads and compiles the OpenCV wasm, which can be slow on a cold cache.
const SSIM_WORKER_BUILD_TIMEOUT_MS = 30_000;
const SSIM_EXECUTE_TIMEOUT_MS = 10_000;
const SSIM_WORKER_QUERY_KEY = ['workers', 'SSIM'];

type SSIMState = {
    shapes: Shape[];
    matches: SSIMMatch[];
    threshold: number;
};

const INITIAL_SSIM_STATE: SSIMState = {
    shapes: [],
    matches: [],
    threshold: 0,
};

export const useSSIMWorker = (enabled = true) => {
    const { data, isLoading, isError, error } = useQuery<{ worker: Worker; instance: Remote<SSIMWorkerInstance> }>({
        queryKey: SSIM_WORKER_QUERY_KEY,
        queryFn: async ({ signal }) => {
            const worker = new Worker(new URL('../../webworkers/ssim-worker', import.meta.url), {
                type: 'module',
            });
            // Wrap in an arrow so `terminate` is called as a method on `worker` (this === worker);
            // passing `worker.terminate` directly would invoke it with `this === signal` → "Illegal invocation".
            signal.addEventListener('abort', () => worker.terminate(), { once: true });

            try {
                const instance = await executeWithTimeout(
                    wrap<SSIMWorkerApi>(worker).build(),
                    i18n.t('annotator.tools.ssim.operations.workerBuild'),
                    SSIM_WORKER_BUILD_TIMEOUT_MS
                );

                if (signal.aborted) {
                    throw signal.reason;
                }

                return { worker, instance };
            } catch (buildError) {
                worker.terminate();

                throw buildError;
            }
        },
        staleTime: Infinity,
        // Tanstack doesn't abort on gc, so evicting the entry would leak the worker; it is
        // terminated by `useTerminateAnnotatorWorkersOnUnmount` instead.
        gcTime: Infinity,
        enabled,
    });

    return { worker: data?.instance, rawWorker: data?.worker, isLoading, isError, error };
};

export const useSSIM = (enabled = true) => {
    const queryClient = useQueryClient();
    const {
        worker: ssim,
        rawWorker,
        isLoading: isLoadingWorker,
        isError: isWorkerError,
        error: workerError,
    } = useSSIMWorker(enabled);

    const [toolState, setToolState] = useState<SSIMState>(INITIAL_SSIM_STATE);
    const [previewThreshold, setPreviewThreshold] = useState<number | null>(null);
    const [ssimProps, setSSIMProps] = useState<RunSSIMProps | null>(null);

    const updateToolState = useCallback(
        (updatedProperties: Partial<SSIMState>, shapeType: SSIMShapeType = 'rectangle') => {
            if (updatedProperties.threshold !== undefined) {
                setPreviewThreshold(null);
            }

            setToolState((currentState) => {
                const newState = { ...currentState, ...updatedProperties };
                const matches = newState.matches.slice(0, newState.threshold);
                const shapes = matches.map(({ shape }) => convertRectToShape(shape, shapeType));

                return {
                    ...newState,
                    shapes,
                };
            });
        },
        []
    );

    const {
        mutate,
        reset: resetMutation,
        isPending,
        error: executionError,
    } = useMutation({
        mutationFn: async (runSSIMProps: RunSSIMProps) => {
            try {
                if (ssim === undefined) {
                    throw new Error('SSIM worker is not initialized yet');
                }

                return await executeWithTimeout(
                    ssim.executeSSIM(toToolRunSSIMProps(runSSIMProps)),
                    i18n.t('annotator.tools.ssim.label'),
                    SSIM_EXECUTE_TIMEOUT_MS
                );
            } catch (executeError) {
                // Comlink calls can't be cancelled and the worker runs them serially, so a stuck (or
                // never built) worker would block every later run: replace it instead.
                rawWorker?.terminate();
                void queryClient.resetQueries({ queryKey: SSIM_WORKER_QUERY_KEY });

                throw executeError;
            }
        },
        onSuccess: (matches, { imageData, existingAnnotations, autoMergeDuplicates, template, shapeType }) => {
            const ssimMatches = convertToolMatchesToGetiMatches(matches);
            const existingRects = autoMergeDuplicates ? getExistingRects(existingAnnotations) : [];
            const filteredMatches = filterSSIMResults(roiFromImage(imageData), ssimMatches, template, existingRects);
            const threshold = guessNumberOfItemsThreshold(filteredMatches);

            updateToolState(
                {
                    matches: filteredMatches,
                    threshold,
                },
                shapeType
            );
        },
        onError: (_error, { template, shapeType }) => {
            updateToolState({ matches: [{ shape: template, confidence: 1 }], threshold: 1 }, shapeType);
        },
    });

    const runSSIM = useCallback(
        (props: RunSSIMProps) => {
            setSSIMProps(props);
            mutate(props);
        },
        [mutate]
    );

    const rerun = useCallback(
        (props: Partial<RunSSIMProps>) => {
            if (ssimProps !== null) {
                runSSIM({ ...ssimProps, ...props });
            }
        },
        [ssimProps, runSSIM]
    );

    const reset = useCallback(() => {
        setToolState(INITIAL_SSIM_STATE);
        setSSIMProps(null);
        setPreviewThreshold(null);
        resetMutation();
    }, [resetMutation]);

    const error = workerError ?? executionError;

    return useMemo(
        () => ({
            runSSIM,
            rerun,
            reset,
            updateToolState,
            toolState,
            previewThreshold,
            setPreviewThreshold,
            isLoading: isLoadingWorker,
            isProcessing: isPending,
            isError: isWorkerError || executionError !== null,
            error,
            worker: ssim as Remote<SSIMWorkerInstance> | undefined,
        }),
        [
            runSSIM,
            rerun,
            reset,
            updateToolState,
            toolState,
            previewThreshold,
            isLoadingWorker,
            isPending,
            isWorkerError,
            executionError,
            error,
            ssim,
        ]
    );
};
