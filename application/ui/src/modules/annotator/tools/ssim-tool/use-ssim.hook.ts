// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useRef, useState } from 'react';

import { i18n } from '@/i18n';
import { roiFromImage } from '@geti-ui/smart-tools/utils';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';
import { clamp } from 'lodash-es';

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

type SSIMRun = {
    props: RunSSIMProps;
    runId: number;
    keepNumberOfItems: boolean;
};

const withShapes = (state: Omit<SSIMState, 'shapes'>, shapeType: SSIMShapeType): SSIMState => ({
    ...state,
    shapes: state.matches.slice(0, state.threshold).map(({ shape }) => convertRectToShape(shape, shapeType)),
});

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
        // The worker is project-agnostic and lives for the whole page; evicting the entry would leak
        // it, since tanstack doesn't abort on gc.
        gcTime: Infinity,
        enabled,
    });

    return { worker: data?.instance, rawWorker: data?.worker, isLoading, isError, error };
};

export const useSSIM = (shapeType: SSIMShapeType, enabled = true) => {
    const queryClient = useQueryClient();
    const {
        worker: ssim,
        rawWorker,
        isLoading: isLoadingWorker,
        isError: isWorkerError,
        error: workerError,
    } = useSSIMWorker(enabled);

    const [toolState, setToolState] = useState<SSIMState>(INITIAL_SSIM_STATE);
    const [ssimProps, setSSIMProps] = useState<RunSSIMProps | null>(null);
    // Results of a run that was reset or superseded must not repopulate the state.
    const runIdRef = useRef(0);

    const updateToolState = useCallback(
        (updatedProperties: Partial<SSIMState>) => {
            setToolState((currentState) => withShapes({ ...currentState, ...updatedProperties }, shapeType));
        },
        [shapeType]
    );

    const {
        mutate,
        reset: resetMutation,
        isPending,
        error: executionError,
    } = useMutation({
        mutationFn: async ({ props, runId }: SSIMRun) => {
            try {
                if (ssim === undefined) {
                    throw new Error('SSIM worker is not initialized yet');
                }

                return await executeWithTimeout(
                    ssim.executeSSIM(toToolRunSSIMProps(props)),
                    i18n.t('annotator.tools.ssim.label'),
                    SSIM_EXECUTE_TIMEOUT_MS
                );
            } catch (executeError) {
                // Comlink calls can't be cancelled and the worker runs them serially, so a stuck (or
                // never built) worker would block every later run: replace it instead.
                // A stale run must not tear down the worker a newer run is using.
                if (runId === runIdRef.current) {
                    rawWorker?.terminate();
                    void queryClient.resetQueries({ queryKey: SSIM_WORKER_QUERY_KEY });
                }

                throw executeError;
            }
        },
        onSuccess: (matches, { props, runId, keepNumberOfItems }) => {
            if (runId !== runIdRef.current) {
                return;
            }

            const { imageData, existingAnnotations, autoMergeDuplicates, template } = props;
            const ssimMatches = convertToolMatchesToGetiMatches(matches);
            const existingRects = autoMergeDuplicates ? getExistingRects(existingAnnotations) : [];
            const filteredMatches = filterSSIMResults(roiFromImage(imageData), ssimMatches, template, existingRects);

            setToolState((currentState) => {
                const threshold = keepNumberOfItems
                    ? clamp(currentState.threshold, 1, filteredMatches.length)
                    : guessNumberOfItemsThreshold(filteredMatches);

                return withShapes({ matches: filteredMatches, threshold }, shapeType);
            });
        },
        onError: (_error, { props, runId }) => {
            if (runId !== runIdRef.current) {
                return;
            }

            updateToolState({ matches: [{ shape: props.template, confidence: 1 }], threshold: 1 });
        },
    });

    const run = useCallback(
        (props: RunSSIMProps, keepNumberOfItems: boolean) => {
            runIdRef.current += 1;
            setSSIMProps(props);
            mutate({ props, runId: runIdRef.current, keepNumberOfItems });
        },
        [mutate]
    );

    const runSSIM = useCallback(
        (props: RunSSIMProps) => {
            // A new template replaces the pending preview instead of being merged into it.
            setToolState(INITIAL_SSIM_STATE);
            run(props, false);
        },
        [run]
    );

    // Keeps the current preview on screen until the new results arrive.
    const rerun = useCallback(
        (props: Partial<RunSSIMProps>) => {
            if (ssimProps !== null) {
                run({ ...ssimProps, ...props }, true);
            }
        },
        [ssimProps, run]
    );

    const reset = useCallback(() => {
        runIdRef.current += 1;
        setToolState(INITIAL_SSIM_STATE);
        setSSIMProps(null);
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
            isLoadingWorker,
            isPending,
            isWorkerError,
            executionError,
            error,
            ssim,
        ]
    );
};
