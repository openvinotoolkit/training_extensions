// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type LogEntry } from '../src/components/training-logs/log-types';

export const getMockedLogEntry = (overrides: Partial<LogEntry['record']> = {}): LogEntry => ({
    text: overrides.message ?? 'Test log message',
    record: {
        elapsed: { repr: '0:01:00.000000', seconds: 60 },
        exception: null,
        extra: {},
        file: { name: 'test_module.py', path: '/app/test_module.py' },
        function: 'test_function',
        level: { icon: 'ℹ️', name: 'INFO', no: 20 },
        line: 42,
        message: 'Test log message',
        module: 'test_module',
        name: 'test_logger',
        process: { id: 1, name: 'MainProcess' },
        thread: { id: 1, name: 'MainThread' },
        time: { repr: '2026-02-06 10:30:00.000000+00:00', timestamp: 1770328200.0 },
        ...overrides,
    },
});

export const getMockedLogEntryJson = (overrides: Partial<LogEntry['record']> = {}): string => {
    return JSON.stringify(getMockedLogEntry(overrides));
};
