// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useSSE } from '../../../hooks/use-sse.hook';
import { type LogEntry } from '../log-types';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

type UseStreamJobLogsReturn = {
    logs: LogEntry[];
    connectionStatus: ConnectionStatus;
};

export const useStreamJobLogs = (jobId: string | undefined, options: { retry: boolean }): UseStreamJobLogsReturn => {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');

    useSSE<LogEntry>(jobId ? `/api/jobs/${jobId}/logs` : undefined, {
        retry: options.retry,
        onMessage: (entry) => {
            setLogs((prev) => [...prev, entry]);
        },
        onOpen: () => {
            setConnectionStatus('connected');
        },
        onRetry: () => {
            setConnectionStatus('connecting');
        },
        onError: () => {
            setConnectionStatus('error');
        },
        onClose: () => {
            setConnectionStatus('disconnected');
        },
    });

    return { logs, connectionStatus };
};
