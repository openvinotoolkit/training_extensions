// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import type {
    IntelligentScissorsWorkerApi,
    IntelligentScissorsWorkerInstance,
} from '../../webworkers/intelligent-scissors.worker.interface';

type IntelligentScissorsRemoteInstance = Remote<IntelligentScissorsWorkerInstance>;

export const useIntelligentScissorsWorker = (enabled = true) => {
    const { data, isLoading, isSuccess, isError } = useQuery<{
        worker: Worker;
        instance: IntelligentScissorsRemoteInstance;
    }>({
        queryKey: ['workers', 'INTELLIGENT_SCISSORS'],
        queryFn: async ({ signal }) => {
            const worker = new Worker(new URL('../../webworkers/intelligent-scissors.worker', import.meta.url), {
                type: 'module',
            });
            // Wrap in an arrow so `terminate` is called as a method on `worker` (this === worker);
            // passing `worker.terminate` directly would invoke it with `this === signal` → "Illegal invocation".
            signal.addEventListener('abort', () => worker.terminate(), { once: true });

            try {
                const instance = await wrap<IntelligentScissorsWorkerApi>(worker).build();

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
        // Tanstack doesn't abort on gc, so evicting the entry would leak the worker; it is
        // terminated by `useTerminateAnnotatorWorkersOnUnmount` instead.
        gcTime: Infinity,
        enabled,
    });

    return { worker: data?.instance, isLoading, isSuccess, isError };
};
