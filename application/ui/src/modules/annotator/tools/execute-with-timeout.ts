// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { i18n } from '@/i18n';

const DEFAULT_WORKER_TIMEOUT_MS = 8000;

export class OperationTimeoutError extends Error {
    name = 'OperationTimeoutError';
}

export const executeWithTimeout = async <T>(
    promise: Promise<T>,
    operation: string,
    timeoutMs = DEFAULT_WORKER_TIMEOUT_MS
): Promise<T> => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    try {
        return await Promise.race([
            promise,
            new Promise<T>((_, reject) => {
                timeoutId = setTimeout(() => {
                    const message = i18n.isInitialized
                        ? i18n.t('annotator.tools.autoSegmentation.timeoutError', { operation, timeoutMs })
                        : `${operation} timed out after ${timeoutMs}ms. Please try refreshing the page.`;
                    reject(new OperationTimeoutError(message));
                }, timeoutMs);
            }),
        ]);
    } finally {
        if (timeoutId !== null) {
            clearTimeout(timeoutId);
        }
    }
};
