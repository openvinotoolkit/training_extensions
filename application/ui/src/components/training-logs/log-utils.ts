// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { LOG_LEVEL_PRIORITY, type LogEntry, type LogLevel } from './log-types';

const isLogEntry = (data: unknown): data is LogEntry => {
    const entry = data as LogEntry;
    return Boolean(entry?.record?.level && entry.record.message !== undefined);
};

const createPlainTextEntry = (text: string): LogEntry => ({
    text,
    record: {
        elapsed: { repr: '', seconds: 0 },
        exception: null,
        extra: {},
        file: { name: '', path: '' },
        function: '',
        level: { icon: '', name: 'INFO', no: 20 },
        line: 0,
        message: text,
        module: '',
        name: '',
        process: { id: 0, name: '' },
        thread: { id: 0, name: '' },
        time: { repr: '', timestamp: Date.now() / 1000 },
    },
});

const toLogEntry = (data: unknown): LogEntry | null => {
    if (isLogEntry(data)) {
        return data;
    }

    return null;
};

export const parseLogLine = (line: string): LogEntry | null => {
    const trimmed = line.trim();

    if (!trimmed) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed) as unknown;

        return toLogEntry(parsed) ?? createPlainTextEntry(trimmed);
    } catch {
        // If the line is not a valid JSON, treat it as plain text
        return createPlainTextEntry(trimmed);
    }
};

export const filterLogs = (logs: LogEntry[], minLevel: LogLevel, searchQuery: string): LogEntry[] => {
    const minPriority = LOG_LEVEL_PRIORITY[minLevel];
    const query = searchQuery.toLowerCase();

    return logs.filter((log) => {
        if (LOG_LEVEL_PRIORITY[log.record.level.name] < minPriority) {
            return false;
        }

        if (query && !log.record.message.toLowerCase().includes(query)) {
            return false;
        }

        return true;
    });
};
