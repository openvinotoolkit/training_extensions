// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useEffectEvent, useRef } from 'react';

import { connectSSE, type SSEOptions } from '@/api';

type SSEConnection = {
    close: () => void;
};

export const useSSE = <T>(url: string | undefined, options: SSEOptions<T>): SSEConnection => {
    const connectionRef = useRef<ReturnType<typeof connectSSE<T>> | null>(null);
    const onMessage = useEffectEvent((data: T) => options.onMessage(data));
    const onError = useEffectEvent((error: Event) => options.onError?.(error));
    const onOpen = useEffectEvent(() => options.onOpen?.());
    const onClose = useEffectEvent(() => options.onClose?.());
    const onRetry = useEffectEvent(() => options.onRetry?.());
    const getRetry = useEffectEvent(() => options.retry);

    useEffect(() => {
        if (!url) {
            return;
        }

        const connection = connectSSE<T>(url, {
            onMessage,
            onError,
            onOpen,
            onClose,
            onRetry,
            retry: getRetry(),
        });

        connectionRef.current = connection;

        return () => {
            connection.close();
            connectionRef.current = null;
        };
    }, [url]);

    const close = useCallback(() => {
        connectionRef.current?.close();
    }, []);

    return { close };
};
