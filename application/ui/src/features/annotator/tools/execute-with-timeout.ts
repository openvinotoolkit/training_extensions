// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { i18n } from '@/i18n';

const DEFAULT_WORKER_TIMEOUT_MS = 8000;

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
                    reject(
                        new Error(i18n.t('annotator.tools.autoSegmentation.timeoutError', { operation, timeoutMs }))
                    );
                }, timeoutMs);
            }),
        ]);
    } finally {
        if (timeoutId !== null) {
            clearTimeout(timeoutId);
        }
    }
};
